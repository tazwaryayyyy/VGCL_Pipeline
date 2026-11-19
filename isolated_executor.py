"""isolated_executor.py

Centralized, OS/container-isolated candidate execution engine for VGCL Phase 0.

Guarantees:
  1. Untrusted code (candidate solutions, reference solutions, test generators)
     NEVER executes in the parent process.
  2. Separate execution contexts:
     - test_gen, solve, and solve_brute run in separate sandbox invocations;
       are_results_equal() executes only in the trusted parent process.
     - Each sandbox runs a non-dumpable supervisor (PID 1) over closed FDs
       spawning an isolated worker (PID 2) so candidate os.write/os._exit
       cannot forge supervisor responses.
  3. Compliant OS/runtime backends (fail-closed):
     - ContainerSandbox: Hardened OCI container (--network none, --read-only,
       --cap-drop ALL, --security-opt no-new-privileges, --ipc none,
       --user 65534:65534, --memory-swap, --pids-limit, --ulimit, --pull=never,
       zero host bind mounts, forced rm -f teardown on timeout).
     - LinuxNamespaceSandbox: Unprivileged user/net/pid/mount namespaces,
       pivot_root RO rootfs, private /proc, private tmpfs /tmp, PR_SET_PDEATHSIG,
       PR_SET_DUMPABLE=0, setrlimit, and seccomp-bpf filtering.
  4. Lossless, bounded JSON IPC over stdin/stdout (no pickle, no shared
     response file, non-finite JSON rejected, strict schema validation).
  5. Fail-closed policy: raises SandboxUnavailableError /
     SandboxInfrastructureError when isolation cannot be established; process-only
     backends are never accepted for untrusted code.
"""

import ast
import base64
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

from quality_gates import (
    _max_dimension,
    case_within_phase_a_policy,
    derive_phase_a_scale_policy,
    extract_declared_constraints,
)

# IPC payloads are already bounded by explicit byte/node budgets, so lift CPython's
# default 4300-digit int<->str conversion limit to keep large ints lossless.
if hasattr(sys, "set_int_max_str_digits"):
    sys.set_int_max_str_digits(1_000_000)

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class SandboxInfrastructureError(RuntimeError):
    """Raised when the sandbox infrastructure fails to establish its security boundary.

    This represents an operational/environment failure, NEVER a candidate capability failure.
    """
    pass


class SandboxUnavailableError(SandboxInfrastructureError):
    """Raised when the requested sandbox backend is unavailable or not supported."""
    pass


# ---------------------------------------------------------------------------
# Strict Equality Helper (Preserved across VGCL)
# ---------------------------------------------------------------------------


def are_results_equal(a: Any, b: Any, rel_tol: float = 1e-6, abs_tol: float = 1e-6) -> bool:
    """Type-aware differential equality check with NumPy scalar unwrapping,

    sequence coercion, and strict bool/int distinction.
    """
    if hasattr(a, "item"):
        try:
            a = a.item()
        except Exception:
            pass
    if hasattr(b, "item"):
        try:
            b = b.item()
        except Exception:
            pass

    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return False
        return all(are_results_equal(x, y, rel_tol, abs_tol) for x, y in zip(a, b))

    if type(a) is not type(b):
        if isinstance(a, bool) or isinstance(b, bool):
            return False
        if not (isinstance(a, (int, float)) and isinstance(b, (int, float))):
            return False

    if isinstance(a, float) or isinstance(b, float):
        if math.isnan(a) and math.isnan(b):
            return True
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)

    if isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            return False
        return all(are_results_equal(a[k], b[k], rel_tol, abs_tol) for k in a)

    return a == b


# ---------------------------------------------------------------------------
# Constants & Lossless Tagged IPC Codec
# ---------------------------------------------------------------------------

ENV_SANDBOX_BACKEND = "VGCL_SANDBOX_BACKEND"
ENV_SANDBOX_IMAGE = "VGCL_SANDBOX_IMAGE"
ENV_CONTAINER_RUNTIME = "VGCL_CONTAINER_RUNTIME"

# Immutable digest pin, verified via `docker pull` + `docker image inspect`
# (RepoDigests) on 2026-11-18. Mutable tags are rejected by ContainerSandbox.
DEFAULT_CONTAINER_IMAGE = "python@sha256:5e2bdb1b0d542d5039cffb8181c65930093949194a65c365dfb1b0fe13a4fa5b"
MAX_IPC_REQUEST_BYTES = 1024 * 1024
MAX_IPC_RESPONSE_BYTES = 262144
MAX_TESTGEN_LARGE_BYTES = 16 * 1024 * 1024
MAX_IPC_STDERR_BYTES = 4096
MAX_IPC_DEPTH = 32
MAX_IPC_NODES = 200000
MAX_IPC_LARGE_NODES = 1000000

_IMAGE_REF_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._/@:+-]{0,255}$")
_IMAGE_DIGEST_RE = re.compile(r"@sha256:[0-9a-f]{64}$")
_RUNTIME_OPT_RE = re.compile(r"^[a-zA-Z0-9._-]{1,64}$")

def _truncate_repr(obj: Any, max_len: int = 500) -> str:
    try:
        r = repr(obj)
        return r[:max_len] + "... [TRUNCATED]" if len(r) > max_len else r
    except Exception:
        return "<Unrepresentable Object>"

def _encode_ipc_value(val: Any, depth: int = 0) -> Any:
    if depth > MAX_IPC_DEPTH:
        raise ValueError("Maximum IPC nesting depth exceeded")
    if hasattr(val, "item") and not isinstance(val, (dict, list, tuple, set, frozenset, bytes, str)):
        try:
            val = val.item()
        except Exception:
            pass
    if val is None or isinstance(val, (bool, int, str)):
        return val
    if isinstance(val, float):
        if math.isnan(val): return {"__vgcl_type__": "float", "val": "nan"}
        if math.isinf(val): return {"__vgcl_type__": "float", "val": "inf" if val > 0 else "-inf"}
        return val
    if isinstance(val, bytes):
        return {"__vgcl_type__": "bytes", "b64": base64.b64encode(val).decode("ascii")}
    if isinstance(val, list):
        return [_encode_ipc_value(x, depth + 1) for x in val]
    if isinstance(val, tuple):
        return {"__vgcl_type__": "tuple", "items": [_encode_ipc_value(x, depth + 1) for x in val]}
    if isinstance(val, set):
        return {"__vgcl_type__": "set", "items": [_encode_ipc_value(x, depth + 1) for x in val]}
    if isinstance(val, frozenset):
        return {"__vgcl_type__": "frozenset", "items": [_encode_ipc_value(x, depth + 1) for x in val]}
    if isinstance(val, dict):
        return {"__vgcl_type__": "dict", "items": [[_encode_ipc_value(k, depth + 1), _encode_ipc_value(v, depth + 1)] for k, v in val.items()]}
    raise TypeError(f"Unsupported non-serializable value type: {type(val).__name__}")

def _decode_ipc_value(obj: Any, depth: int = 0, budget: Optional[List[int]] = None) -> Any:
    if budget is None:
        budget = [MAX_IPC_NODES]
    budget[0] -= 1
    if budget[0] < 0:
        raise ValueError("IPC node budget exceeded")
    if depth > MAX_IPC_DEPTH:
        raise ValueError("IPC nesting depth exceeded")
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, float):
        if not math.isfinite(obj): raise ValueError("Non-finite float in IPC")
        return obj
    if isinstance(obj, list):
        return [_decode_ipc_value(x, depth + 1, budget) for x in obj]
    if isinstance(obj, dict):
        tag = obj.get("__vgcl_type__")
        if tag == "float":
            v = obj.get("val")
            if v in ("nan", "inf", "-inf"): return float(v)
            raise ValueError(f"Invalid tagged float: {v!r}")
        if tag == "bytes":
            if not isinstance(obj.get("b64"), str): raise ValueError("Invalid bytes tag")
            return base64.b64decode(obj["b64"].encode("ascii"), validate=True)
        if tag in ("tuple", "set", "frozenset"):
            items = obj.get("items")
            if not isinstance(items, list): raise ValueError(f"Invalid {tag} tag")
            vals = [_decode_ipc_value(x, depth + 1, budget) for x in items]
            return tuple(vals) if tag == "tuple" else (set(vals) if tag == "set" else frozenset(vals))
        if tag == "dict":
            items = obj.get("items")
            if not isinstance(items, list): raise ValueError("Invalid dict tag")
            out = {}
            for pair in items:
                if not (isinstance(pair, list) and len(pair) == 2): raise ValueError("Invalid dict pair")
                k = _decode_ipc_value(pair[0], depth + 1, budget)
                v = _decode_ipc_value(pair[1], depth + 1, budget)
                try:
                    if k in out: raise ValueError("Duplicate dict key in IPC")
                    out[k] = v
                except TypeError:
                    raise ValueError("Unhashable dict key in IPC")
            return out
        raise ValueError(f"Unknown __vgcl_type__: {tag!r}")
    raise ValueError(f"Disallowed IPC value type: {type(obj).__name__}")

def _reject_nonfinite_json(token: str) -> Any:
    raise ValueError(f"Non-finite JSON constant rejected: {token!r}")

SENSITIVE_ENV_PREFIXES = (
    "GEMINI", "GOOGLE", "OPENAI", "ANTHROPIC", "HF_", "HUGGING", "KAGGLE",
    "AWS", "GITHUB", "SSH", "TOKEN", "SECRET", "KEY", "PASS", "AUTH", "VGCL_",
)

def build_sanitized_environment() -> Dict[str, str]:
    return {
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "HOME": "/tmp",
        "TMPDIR": "/tmp",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONUNBUFFERED": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
    }

INNER_WORKER_CODE = r"""
import ast, base64, copy, ctypes, errno, inspect, json, math, os, signal, struct, sys, time
MAX_IPC_DEPTH = 32
MAX_LARGE_REQUEST_BYTES = 16 * 1024 * 1024
def truncate_repr(obj, max_len=500):
    try:
        r = repr(obj)
        return r[:max_len] + "... [TRUNCATED]" if len(r) > max_len else r
    except Exception:
        return "<Unrepresentable Object>"
def get_peak_memory_mb():
    try:
        import resource
        ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return ru / (1024.0 * 1024.0) if sys.platform == "darwin" else ru / 1024.0
    except Exception:
        return 0.0
def encode_val(val, depth=0):
    if depth > MAX_IPC_DEPTH: raise ValueError("Maximum nesting depth exceeded")
    if hasattr(val, "item") and not isinstance(val, (dict, list, tuple, set, frozenset, bytes, str)):
        try: val = val.item()
        except Exception: pass
    if val is None or isinstance(val, (bool, int, str)): return val
    if isinstance(val, float):
        if math.isnan(val): return {"__vgcl_type__": "float", "val": "nan"}
        if math.isinf(val): return {"__vgcl_type__": "float", "val": "inf" if val > 0 else "-inf"}
        return val
    if isinstance(val, bytes): return {"__vgcl_type__": "bytes", "b64": base64.b64encode(val).decode("ascii")}
    if isinstance(val, list): return [encode_val(x, depth + 1) for x in val]
    if isinstance(val, tuple): return {"__vgcl_type__": "tuple", "items": [encode_val(x, depth + 1) for x in val]}
    if isinstance(val, set): return {"__vgcl_type__": "set", "items": [encode_val(x, depth + 1) for x in val]}
    if isinstance(val, frozenset): return {"__vgcl_type__": "frozenset", "items": [encode_val(x, depth + 1) for x in val]}
    if isinstance(val, dict): return {"__vgcl_type__": "dict", "items": [[encode_val(k, depth + 1), encode_val(v, depth + 1)] for k, v in val.items()]}
    raise TypeError(f"Unsupported non-serializable value type: {type(val).__name__}")
def decode_val(obj, depth=0):
    if depth > MAX_IPC_DEPTH: raise ValueError("Maximum nesting depth exceeded")
    if obj is None or isinstance(obj, (bool, int, float, str)): return obj
    if isinstance(obj, list): return [decode_val(x, depth + 1) for x in obj]
    if isinstance(obj, dict):
        tag = obj.get("__vgcl_type__")
        if tag == "float": return float(obj["val"])
        if tag == "bytes": return base64.b64decode(obj["b64"].encode("ascii"), validate=True)
        if tag == "tuple": return tuple(decode_val(x, depth + 1) for x in obj["items"])
        if tag == "set": return set(decode_val(x, depth + 1) for x in obj["items"])
        if tag == "frozenset": return frozenset(decode_val(x, depth + 1) for x in obj["items"])
        if tag == "dict": return {decode_val(k, depth + 1): decode_val(v, depth + 1) for k, v in obj["items"]}
    raise ValueError("Invalid encoded value in worker request")
def build_seccomp_insns(arch, clone_nr, clone3_nr, blocked):
    RET_ALLOW, RET_EPERM, RET_ENOSYS = 0x7FFF0000, 0x00050000 | errno.EPERM, 0x00050000 | errno.ENOSYS
    insns = [
        struct.pack("HBBI", 0x20, 0, 0, 4),
        struct.pack("HBBI", 0x15, 1, 0, arch),
        struct.pack("HBBI", 0x06, 0, 0, RET_EPERM),
        struct.pack("HBBI", 0x20, 0, 0, 0),
    ]
    for nr in blocked:
        insns.append(struct.pack("HBBI", 0x15, 0, 1, nr))
        insns.append(struct.pack("HBBI", 0x06, 0, 0, RET_EPERM))
    insns.extend([
        struct.pack("HBBI", 0x15, 0, 1, clone3_nr),
        struct.pack("HBBI", 0x06, 0, 0, RET_ENOSYS),
        struct.pack("HBBI", 0x15, 0, 5, clone_nr),
        struct.pack("HBBI", 0x20, 0, 0, 16),
        struct.pack("HBBI", 0x54, 0, 0, 0x7E020000),
        struct.pack("HBBI", 0x15, 0, 1, 0),
        struct.pack("HBBI", 0x06, 0, 0, RET_ALLOW),
        struct.pack("HBBI", 0x06, 0, 0, RET_EPERM),
        struct.pack("HBBI", 0x06, 0, 0, RET_ALLOW),
    ])
    return insns
def apply_inner_isolation(payload):
    if sys.platform.startswith("linux"):
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(1, signal.SIGKILL, 0, 0, 0) != 0: raise OSError(ctypes.get_errno(), "PR_SET_PDEATHSIG failed")
        if libc.prctl(4, 0, 0, 0, 0) != 0: raise OSError(ctypes.get_errno(), "PR_SET_DUMPABLE failed")
    if payload.get("apply_rlimits"):
        import resource
        mem_b = int(payload.get("memory_limit_mb", 512)) * 1024 * 1024
        cpu_s = int(payload.get("cpu_limit_s", 6))
        resource.setrlimit(resource.RLIMIT_AS, (mem_b, mem_b))
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s))
        resource.setrlimit(resource.RLIMIT_FSIZE, (1024 * 1024, 1024 * 1024))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        _, hn = resource.getrlimit(resource.RLIMIT_NOFILE)
        nf = min(64, hn) if hn > 0 else 64
        resource.setrlimit(resource.RLIMIT_NOFILE, (nf, nf))
        np_lim = payload.get("nproc_limit")
        if isinstance(np_lim, int) and np_lim > 0:
            _, hnp = resource.getrlimit(resource.RLIMIT_NPROC)
            nt = min(np_lim, hnp) if hnp > 0 else np_lim
            resource.setrlimit(resource.RLIMIT_NPROC, (nt, nt))
    if payload.get("apply_seccomp"):
        import platform
        libc = ctypes.CDLL(None, use_errno=True)
        mach = platform.machine().lower()
        if mach in ("x86_64", "amd64"):
            arch, clone_nr, clone3_nr = 0xC000003E, 56, 435
            blocked = [41, 42, 43, 44, 46, 49, 50, 101, 155, 157, 161, 165, 166, 175, 176, 246, 248, 249, 250, 272, 288, 298, 307, 308, 310, 311, 313, 321, 323, 425, 428, 429, 430, 431, 432, 433, 438, 442]
        elif mach in ("aarch64", "arm64"):
            arch, clone_nr, clone3_nr = 0xC00000B7, 220, 435
            blocked = [39, 40, 41, 51, 97, 104, 105, 106, 117, 167, 198, 200, 201, 202, 203, 206, 211, 217, 218, 219, 241, 242, 268, 269, 270, 271, 273, 280, 282, 425, 428, 429, 430, 431, 432, 433, 438, 442]
        else:
            raise RuntimeError(f"Unsupported seccomp arch: {mach}")
        insns = build_seccomp_insns(arch, clone_nr, clone3_nr, blocked)
        buf = ctypes.create_string_buffer(b"".join(insns))
        class _Fprog(ctypes.Structure): _fields_ = [("len", ctypes.c_ushort), ("filter", ctypes.c_void_p)]
        prog = _Fprog(len(insns), ctypes.addressof(buf))
        if libc.prctl(38, 1, 0, 0, 0) != 0: raise OSError(ctypes.get_errno(), "PR_SET_NO_NEW_PRIVS failed")
        if libc.prctl(22, 2, ctypes.byref(prog)) != 0: raise OSError(ctypes.get_errno(), "PR_SET_SECCOMP failed")
def run_probe_self_check():
    import _socket, resource
    libc = ctypes.CDLL(None, use_errno=True)
    res = {"root_entries": sorted(os.listdir("/")), "proc_pids": [p for p in os.listdir("/proc") if p.isdigit()]}
    try:
        with open("/usr/vgcl_ro_check", "w") as f: f.write("x")
        res["usr_readonly"] = False
    except OSError: res["usr_readonly"] = True
    try:
        p = f"/tmp/vgcl_tmp_{os.getpid()}"
        with open(p, "w") as f: f.write("ok")
        os.remove(p)
        res["tmp_writable"] = True
    except OSError: res["tmp_writable"] = False
    try:
        with open("/proc/1/environ", "rb") as f: p1_env = f.read(8192).upper()
        p1_keys = [e.split(b"=", 1)[0] for e in p1_env.split(b"\0") if e]
        res["proc_1_clean"] = not any(k.startswith((b"GEMINI", b"KAGGLE", b"VGCL_", b"OPENAI")) for k in p1_keys)
    except OSError: res["proc_1_clean"] = True
    try:
        s = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM); s.close()
        res["net_blocked"] = False
    except OSError: res["net_blocked"] = True
    ctypes.set_errno(0)
    rc_p = libc.prctl(1, 0, 0, 0, 0)
    res["prctl_blocked"] = (rc_p != 0 and ctypes.get_errno() == errno.EPERM)
    as_s, _ = resource.getrlimit(resource.RLIMIT_AS)
    co_s, _ = resource.getrlimit(resource.RLIMIT_CORE)
    fs_s, _ = resource.getrlimit(resource.RLIMIT_FSIZE)
    res["rlimits_applied"] = (as_s > 0 and co_s == 0 and fs_s == 1024 * 1024)
    res["env_clean"] = not any(k.upper().startswith(("GEMINI", "KAGGLE", "VGCL_", "OPENAI")) for k in os.environ)
    return {"status": "ok", "probe": res}
def run_test_gen(payload):
    code, seed, scale = payload.get("code", ""), payload.get("seed", 42), payload.get("scale", "small")
    max_cases, max_dim = int(payload.get("max_cases", 10)), payload.get("max_small_dimension")
    try: ast.parse(code)
    except SyntaxError as e: return {"status": "syntax_error", "exception_type": "SyntaxError", "error_msg": str(e)[:1000]}
    g = {}
    try: exec(compile(code, "<test_gen>", "exec"), g)
    except Exception as e: return {"status": "exception", "exception_type": type(e).__name__, "error_msg": str(e)[:1000]}
    gen_fn = g.get("generate_test_cases")
    if not callable(gen_fn): return {"status": "missing_function", "exception_type": "MissingFunction", "error_msg": "solve or generate_test_cases not defined"}
    try:
        try:
            sig = inspect.signature(gen_fn); params = sig.parameters
            has_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
            if max_dim is not None and scale == "small" and ("max_small_dimension" in params or has_kw):
                raw_cases = gen_fn(seed=seed, scale=scale, max_small_dimension=int(max_dim))
            elif max_dim is not None and scale == "small" and len(params) >= 3:
                raw_cases = gen_fn(seed, scale, int(max_dim))
            elif "seed" in params and "scale" in params: raw_cases = gen_fn(seed=seed, scale=scale)
            elif len(params) >= 2: raw_cases = gen_fn(seed, scale)
            elif len(params) == 1: raw_cases = gen_fn(seed)
            else: raw_cases = gen_fn(seed=seed, scale=scale)
        except Exception: raw_cases = gen_fn(seed, scale)
        if not isinstance(raw_cases, list): return {"status": "format_error", "exception_type": "FormatContractError", "error_msg": f"generate_test_cases must return list, got {type(raw_cases).__name__}"}
        return {"status": "ok", "cases": encode_val(raw_cases[:max_cases])}
    except TypeError as te: return {"status": "unsupported_type", "exception_type": "UnsupportedReturnTypeError", "error_msg": str(te)[:1000]}
    except Exception as e: return {"status": "exception", "exception_type": type(e).__name__, "error_msg": f"test_gen failed: {e}"[:1000]}
def run_fn(payload):
    code, fn_name = payload.get("code", ""), payload.get("fn_name", "solve")
    cases = decode_val(payload.get("cases", []))
    phase_b_mode, consec_cases = bool(payload.get("phase_b_mode", False)), decode_val(payload.get("consecutive_cases", []))
    _perf_counter, _deepcopy = time.perf_counter, copy.deepcopy
    try: ast.parse(code)
    except SyntaxError as e: return {"status": "syntax_error", "exception_type": "SyntaxError", "error_msg": str(e)[:1000], "execution_time": 0.0, "memory_mb": 0.0}
    g = {}
    try: exec(compile(code, f"<{fn_name}>", "exec"), g)
    except Exception as e: return {"status": "exception", "fail_idx": 0, "exception_type": type(e).__name__, "error_msg": str(e)[:1000], "execution_time": 0.0, "memory_mb": 0.0}
    fn = g.get(fn_name)
    if not callable(fn): return {"status": "missing_function", "exception_type": "MissingFunction", "error_msg": f"{fn_name} not defined", "execution_time": 0.0, "memory_mb": 0.0}
    consec_pairs = []
    for idx, args in enumerate(consec_cases):
        pa = args if isinstance(args, (list, tuple)) else (args,)
        try: consec_pairs.append([encode_val(fn(*_deepcopy(pa))), encode_val(fn(*_deepcopy(pa)))])
        except TypeError as te: return {"status": "unsupported_type", "fail_idx": idx, "exception_type": "UnsupportedReturnTypeError", "error_msg": str(te)[:1000], "execution_time": 0.0, "memory_mb": get_peak_memory_mb()}
        except Exception as e: return {"status": "exception", "fail_idx": idx, "exception_type": type(e).__name__, "error_msg": str(e)[:1000], "execution_time": 0.0, "memory_mb": get_peak_memory_mb()}
    if phase_b_mode:
        times = []
        for idx, args in enumerate(cases):
            pa = args if isinstance(args, (list, tuple)) else (args,)
            try:
                t0 = _perf_counter(); out = fn(*_deepcopy(pa)); dt = _perf_counter() - t0
                times.append(round(dt, 6))
                if dt > 2.0: return {"status": "timeout_case", "fail_idx": idx, "per_case_times": times, "actual_preview": truncate_repr(out), "execution_time": sum(times), "memory_mb": get_peak_memory_mb()}
            except Exception as e: return {"status": "exception", "fail_idx": idx, "per_case_times": times, "exception_type": type(e).__name__, "error_msg": str(e)[:1000], "execution_time": sum(times), "memory_mb": get_peak_memory_mb()}
        return {"status": "ok", "per_case_times": times, "consecutive_pairs": consec_pairs, "execution_time": sum(times), "memory_mb": get_peak_memory_mb()}
    outputs, t0 = [], _perf_counter()
    for idx, args in enumerate(cases):
        pa = args if isinstance(args, (list, tuple)) else (args,)
        try: outputs.append(encode_val(fn(*_deepcopy(pa))))
        except TypeError as te: return {"status": "unsupported_type", "fail_idx": idx, "outputs": outputs, "exception_type": "UnsupportedReturnTypeError", "error_msg": str(te)[:1000], "execution_time": _perf_counter() - t0, "memory_mb": get_peak_memory_mb()}
        except Exception as e: return {"status": "exception", "fail_idx": idx, "outputs": outputs, "exception_type": type(e).__name__, "error_msg": str(e)[:1000], "execution_time": _perf_counter() - t0, "memory_mb": get_peak_memory_mb()}
    return {"status": "ok", "outputs": outputs, "consecutive_pairs": consec_pairs, "execution_time": _perf_counter() - t0, "memory_mb": get_peak_memory_mb()}
def main():
    os.environ["PATH"] = "/usr/local/bin:/usr/bin:/bin"
    if hasattr(sys, "set_int_max_str_digits"): sys.set_int_max_str_digits(1000000)
    pipe_fd = int(sys.argv[1]); sys.argv = [sys.argv[0]]
    raw_in = sys.stdin.read(MAX_LARGE_REQUEST_BYTES + 1)
    if len(raw_in) > MAX_LARGE_REQUEST_BYTES: sys.exit(2)
    payload = json.loads(raw_in)
    apply_inner_isolation(payload)
    task = payload.get("task")
    try:
        if task == "probe_self_check": res = run_probe_self_check()
        elif task == "test_gen": res = run_test_gen(payload)
        elif task == "run_fn": res = run_fn(payload)
        else: res = {"status": "protocol_error", "error_msg": f"Unknown task: {task}"}
    except Exception as exc: res = {"status": "exception", "fail_idx": 0, "exception_type": type(exc).__name__, "error_msg": str(exc)[:1000], "execution_time": 0.0, "memory_mb": 0.0}
    data = {k: v for k, v in res.items() if k not in ("execution_time", "memory_mb")}
    encoded = json.dumps(data, allow_nan=False).encode("utf-8") + b"\n"
    view, w = memoryview(encoded), 0
    while w < len(encoded): w += os.write(pipe_fd, view[w:])
if __name__ == "__main__": main()
"""

RUNNER_CODE = r"""
import ctypes, json, os, resource, select, signal, subprocess, sys, threading, time
def _read_pipe(fd, max_b, buf, flags):
    try:
        while True:
            chunk = os.read(fd, 4096)
            if not chunk: break
            rem = max_b - len(buf)
            if rem > 0: buf.extend(chunk[:rem])
            if len(chunk) > rem: flags[0] = True; break
            if b"\n" in chunk:
                if len(buf.split(b"\n", 1)[1].strip()) > 0: flags[1] = True
                break
    except Exception: pass
def _reject_constant(token):
    raise ValueError(f"Non-finite JSON constant rejected: {token!r}")
_SEMANTIC_STATUS = {"ok", "syntax_error", "missing_function", "exception", "unsupported_type", "format_error", "timeout_case"}
def main():
    os.environ["PATH"] = "/usr/local/bin:/usr/bin:/bin"
    if hasattr(sys, "set_int_max_str_digits"): sys.set_int_max_str_digits(1000000)
    resp_fd = int(sys.argv[1]); sys.argv = [sys.argv[0]]
    if sys.platform.startswith("linux"):
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(4, 0, 0, 0, 0) != 0: sys.exit(4)
    raw_in = sys.stdin.read(20 * 1024 * 1024 + 4096)
    env_p = json.loads(raw_in, parse_constant=_reject_constant)
    inner_src, nonce = env_p["inner_src"], env_p["nonce"]
    inner_bytes = json.dumps(env_p["payload"], allow_nan=False).encode("utf-8")
    max_resp = int(env_p.get("max_response_bytes", 262144))
    timeout_s = float(env_p.get("timeout_s", 4.0))
    r_fd, w_fd = os.pipe()
    os.set_inheritable(w_fd, True)
    t_start = time.perf_counter()
    proc = subprocess.Popen([sys.executable, "-I", "-c", inner_src, str(w_fd)], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, pass_fds=(w_fd,), close_fds=True, cwd="/tmp")
    os.close(w_fd)
    buf, flags = bytearray(), [False, False]
    t = threading.Thread(target=_read_pipe, args=(r_fd, max_resp + 1, buf, flags), daemon=True)
    t.start()
    timed_out, rc = False, -1
    try:
        if proc.stdin:
            try: proc.stdin.write(inner_bytes); proc.stdin.flush()
            except BrokenPipeError: pass
            finally: proc.stdin.close()
        try: rc = proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True; proc.kill(); rc = proc.wait()
    finally:
        elapsed = round(time.perf_counter() - t_start, 6)
        t.join(timeout=0.3)
        try:
            while True:
                rlist, _, _ = select.select([r_fd], [], [], 0)
                if not rlist: break
                chunk = os.read(r_fd, 4096)
                if not chunk: break
                if chunk.strip(): flags[1] = True
        except Exception: pass
        try: os.close(r_fd)
        except Exception: pass
    mem_mb = round(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024.0, 3)
    if timed_out: out = {"status": "timeout", "exception_type": "TimeoutError", "error_msg": f"Execution timed out (> {timeout_s:.1f}s wall clock)", "execution_time": elapsed, "memory_mb": mem_mb}
    elif flags[0]: out = {"status": "protocol_error", "exception_type": "WorkerProtocolError", "error_msg": "Worker response exceeded size limit", "execution_time": elapsed, "memory_mb": mem_mb}
    elif flags[1]: out = {"status": "protocol_error", "exception_type": "WorkerProtocolError", "error_msg": "Trailing IPC data rejected", "execution_time": elapsed, "memory_mb": mem_mb}
    elif rc != 0:
        is_oom = rc in (-9, 137, -11)
        out = {"status": "crash", "exception_type": "MemoryError" if rc in (-9, 137) else "ProcessCrashError", "error_msg": f"Worker exited unexpectedly (code {rc})", "execution_time": elapsed, "memory_mb": max(mem_mb, 512.0) if is_oom else mem_mb}
    else:
        if b"\n" not in buf: out = {"status": "protocol_error", "exception_type": "WorkerProtocolError", "error_msg": "Truncated worker output", "execution_time": elapsed, "memory_mb": mem_mb}
        else:
            line = bytes(buf).split(b"\n", 1)[0]
            try:
                data = json.loads(line.decode("utf-8"), parse_constant=_reject_constant)
                if not isinstance(data, dict): raise ValueError("Worker data frame must be a JSON object")
                forbidden = {"execution_time", "memory_mb", "passed"} & set(data.keys())
                if forbidden: raise ValueError(f"Candidate must not assert trusted fields: {sorted(forbidden)}")
                if data.get("status") not in _SEMANTIC_STATUS: raise ValueError(f"Invalid candidate status: {data.get('status')!r}")
                out = dict(data); out["execution_time"] = elapsed; out["memory_mb"] = mem_mb
            except Exception as err: out = {"status": "protocol_error", "exception_type": "WorkerProtocolError", "error_msg": f"Malformed worker data: {err}", "execution_time": elapsed, "memory_mb": mem_mb}
    if env_p["payload"].get("task") != "run_fn":
        out.pop("execution_time", None); out.pop("memory_mb", None)
    frame = json.dumps({"nonce": nonce, "response": out}, allow_nan=False).encode("utf-8") + b"\n"
    view, w = memoryview(frame), 0
    while w < len(frame): w += os.write(resp_fd, view[w:])
if __name__ == "__main__": main()
"""

SUPERVISOR_CODE = r"""
import ctypes, json, os, secrets, select, signal, subprocess, sys, threading
def _read_pipe(fd, max_b, buf, flags):
    try:
        while True:
            chunk = os.read(fd, 4096)
            if not chunk: break
            rem = max_b - len(buf)
            if rem > 0: buf.extend(chunk[:rem])
            if len(chunk) > rem: flags[0] = True; break
            if b"\n" in chunk:
                if len(buf.split(b"\n", 1)[1].strip()) > 0: flags[1] = True
                break
    except Exception: pass
def _reject_constant(token):
    raise ValueError(f"Non-finite JSON constant rejected: {token!r}")
def main():
    os.environ["PATH"] = "/usr/local/bin:/usr/bin:/bin"
    if hasattr(sys, "set_int_max_str_digits"): sys.set_int_max_str_digits(1000000)
    if sys.platform.startswith("linux"):
        libc = ctypes.CDLL(None, use_errno=True)
        if "--pdeath" in sys.argv[1:]:
            if libc.prctl(1, signal.SIGKILL, 0, 0, 0) != 0: sys.exit(3)
        if libc.prctl(4, 0, 0, 0, 0) != 0: sys.exit(4)
    max_req = 16 * 1024 * 1024 + 262144
    raw_in = sys.stdin.read(max_req + 1)
    if len(raw_in) > max_req: sys.exit(2)
    env_p = json.loads(raw_in, parse_constant=_reject_constant)
    nonce = secrets.token_hex(16)
    runner_env = {"inner_src": env_p["inner_src"], "payload": env_p["payload"], "nonce": nonce, "max_response_bytes": env_p.get("max_response_bytes", 262144), "timeout_s": env_p.get("timeout_s", 4.0)}
    runner_bytes = json.dumps(runner_env, allow_nan=False).encode("utf-8")
    max_resp = int(env_p.get("max_response_bytes", 262144))
    timeout_s = float(env_p.get("timeout_s", 4.0))
    r_fd, w_fd = os.pipe()
    os.set_inheritable(w_fd, True)
    proc = subprocess.Popen([sys.executable, "-I", "-c", env_p["runner_src"], str(w_fd)], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, pass_fds=(w_fd,), close_fds=True, cwd="/tmp")
    os.close(w_fd)
    buf, flags = bytearray(), [False, False]
    t = threading.Thread(target=_read_pipe, args=(r_fd, max_resp + 1, buf, flags), daemon=True)
    t.start()
    timed_out, rc = False, -1
    try:
        if proc.stdin:
            try: proc.stdin.write(runner_bytes); proc.stdin.flush()
            except BrokenPipeError: pass
            finally: proc.stdin.close()
        try: rc = proc.wait(timeout=timeout_s + 1.0)
        except subprocess.TimeoutExpired:
            timed_out = True; proc.kill(); rc = proc.wait()
    finally:
        t.join(timeout=0.3)
        try:
            while True:
                rlist, _, _ = select.select([r_fd], [], [], 0)
                if not rlist: break
                chunk = os.read(r_fd, 4096)
                if not chunk: break
                if chunk.strip(): flags[1] = True
        except Exception: pass
        try: os.close(r_fd)
        except Exception: pass
    if timed_out: out = {"status": "timeout", "exception_type": "TimeoutError", "error_msg": f"Execution timed out (> {timeout_s:.1f}s wall clock)", "execution_time": timeout_s, "memory_mb": 0.0}
    elif flags[0]: out = {"status": "protocol_error", "exception_type": "WorkerProtocolError", "error_msg": "Worker response exceeded size limit", "execution_time": 0.0, "memory_mb": 0.0}
    elif flags[1]: out = {"status": "protocol_error", "exception_type": "WorkerProtocolError", "error_msg": "Trailing IPC data rejected", "execution_time": 0.0, "memory_mb": 0.0}
    elif rc != 0:
        out = {"status": "crash", "exception_type": "ProcessCrashError", "error_msg": f"Trusted runner exited unexpectedly (code {rc})", "execution_time": 0.0, "memory_mb": 0.0}
    else:
        if b"\n" not in buf: out = {"status": "protocol_error", "exception_type": "WorkerProtocolError", "error_msg": "Truncated worker output", "execution_time": 0.0, "memory_mb": 0.0}
        else:
            line = bytes(buf).split(b"\n", 1)[0]
            try:
                frame = json.loads(line.decode("utf-8"), parse_constant=_reject_constant)
                if not isinstance(frame, dict) or not secrets.compare_digest(str(frame.get("nonce", "")), nonce): raise ValueError("Untrusted response origin rejected")
                out = frame.get("response")
                if not isinstance(out, dict): raise ValueError("Trusted runner response must be a JSON object")
            except Exception as err: out = {"status": "protocol_error", "exception_type": "WorkerProtocolError", "error_msg": f"Malformed worker JSON: {err}", "execution_time": 0.0, "memory_mb": 0.0}
    if env_p["payload"].get("task") != "run_fn":
        out.pop("execution_time", None); out.pop("memory_mb", None)
    sys.stdout.write(json.dumps(out, allow_nan=False))
    sys.stdout.flush()
if __name__ == "__main__": main()
"""

def _read_bounded(stream: Any, max_bytes: int, out_buf: bytearray, overflow_flag: List[bool]) -> None:
    try:
        while True:
            chunk = stream.read(4096)
            if not chunk: break
            rem = max_bytes - len(out_buf)
            if rem > 0: out_buf.extend(chunk[:rem])
            if len(chunk) > rem:
                overflow_flag[0] = True
                while stream.read(8192): pass
                break
    except Exception: pass
    finally:
        try: stream.close()
        except Exception: pass

def _communicate_bounded(proc: subprocess.Popen, input_bytes: bytes, timeout_s: float, max_stdout_bytes: int, kill_fn: Any) -> Tuple[bool, bytes, bytes, int]:
    stdout_buf, stderr_buf = bytearray(), bytearray()
    out_ov, err_ov = [False], [False]
    t_out = threading.Thread(target=_read_bounded, args=(proc.stdout, max_stdout_bytes, stdout_buf, out_ov), daemon=True)
    t_err = threading.Thread(target=_read_bounded, args=(proc.stderr, MAX_IPC_STDERR_BYTES, stderr_buf, err_ov), daemon=True)
    t_out.start(); t_err.start()
    timed_out, rc = False, -1
    try:
        if proc.stdin is not None:
            try: proc.stdin.write(input_bytes); proc.stdin.flush()
            except BrokenPipeError: pass
            finally:
                try: proc.stdin.close()
                except Exception: pass
        try: rc = proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True; kill_fn()
            try: rc = proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                try: proc.kill(); rc = proc.wait(timeout=1.0)
                except Exception: rc = -9
    finally:
        t_out.join(timeout=1.0); t_err.join(timeout=1.0)
        for s in (proc.stdin, proc.stdout, proc.stderr):
            if s is not None:
                try: s.close()
                except Exception: pass
    if timed_out: return True, b"", bytes(stderr_buf), -9
    if out_ov[0]: raise ValueError("Worker response exceeded size limit")
    return False, bytes(stdout_buf), bytes(stderr_buf), rc

_ALLOWED_KEYS_BY_TASK = {
    "probe_self_check": {"status", "probe", "exception_type", "error_msg"},
    "test_gen": {"status", "cases", "exception_type", "error_msg"},
    "run_fn": {"status", "outputs", "consecutive_pairs", "per_case_times", "actual_preview", "fail_idx", "exception_type", "error_msg", "execution_time", "memory_mb"},
}

_ALLOWED_STATUS_BY_TASK = {
    "probe_self_check": {"ok", "exception", "protocol_error", "timeout", "crash"},
    "test_gen": {"ok", "syntax_error", "exception", "missing_function", "format_error", "unsupported_type", "protocol_error", "timeout", "crash"},
    "run_fn": {"ok", "syntax_error", "missing_function", "exception", "unsupported_type", "timeout", "timeout_case", "protocol_error", "crash"},
}

def _validate_worker_response(raw_bytes: bytes, task: str, max_nodes: int = MAX_IPC_NODES) -> Dict[str, Any]:
    data = json.loads(raw_bytes.decode("utf-8"), parse_constant=_reject_nonfinite_json)
    if not isinstance(data, dict): raise ValueError("Worker response must be a JSON object")
    allowed = _ALLOWED_KEYS_BY_TASK.get(task)
    if allowed is None: raise ValueError(f"Unknown task schema: {task}")
    extra = set(data.keys()) - allowed
    if extra: raise ValueError(f"Unexpected keys in worker response: {sorted(extra)}")
    if "passed" in data: raise ValueError("Untrusted worker must not assert 'passed'")
    if not isinstance(data.get("status"), str) or not data["status"]: raise ValueError("Missing/invalid 'status'")
    if data["status"] not in _ALLOWED_STATUS_BY_TASK[task]: raise ValueError(f"Unknown status for task '{task}': {data['status']!r}")
    if data["status"] == "ok":
        if task == "probe_self_check" and not isinstance(data.get("probe"), dict): raise ValueError("probe_self_check ok requires 'probe' object")
        if task == "test_gen" and "cases" not in data: raise ValueError("test_gen ok requires 'cases'")
        if task == "run_fn":
            if "outputs" not in data and "per_case_times" not in data: raise ValueError("run_fn ok requires 'outputs' or 'per_case_times'")
            if "execution_time" not in data or "memory_mb" not in data: raise ValueError("run_fn ok requires 'execution_time' and 'memory_mb'")
    for sf, ml in (("exception_type", 128), ("error_msg", 2000), ("actual_preview", 600)):
        if sf in data and data[sf] is not None:
            if not isinstance(data[sf], str) or len(data[sf]) > ml: raise ValueError(f"Invalid field '{sf}'")
    for ff in ("execution_time", "memory_mb"):
        if ff in data:
            v = data[ff]
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0: raise ValueError(f"Invalid numeric '{ff}'")
    if "fail_idx" in data:
        fi = data["fail_idx"]
        if isinstance(fi, bool) or not isinstance(fi, int) or fi < 0: raise ValueError("Invalid 'fail_idx'")
    if "per_case_times" in data:
        pct = data["per_case_times"]
        if not isinstance(pct, list) or any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or x < 0 for x in pct):
            raise ValueError("Invalid 'per_case_times'")
    if "cases" in data:
        data["cases"] = _decode_ipc_value(data["cases"], budget=[max_nodes])
        if not isinstance(data["cases"], list): raise ValueError("'cases' must decode to list")
    if "outputs" in data:
        data["outputs"] = _decode_ipc_value(data["outputs"], budget=[max_nodes])
        if not isinstance(data["outputs"], list): raise ValueError("'outputs' must decode to list")
    if "consecutive_pairs" in data:
        pairs = _decode_ipc_value(data["consecutive_pairs"], budget=[max_nodes])
        if not isinstance(pairs, list) or any(not (isinstance(p, list) and len(p) == 2) for p in pairs): raise ValueError("Invalid 'consecutive_pairs'")
        data["consecutive_pairs"] = pairs
    return data

_LINUX_STATIC_MOUNT_SH = r"""
set -e
ROOTFS="$1"; PYEXE="$2"; PYBASE="$3"; PYVENV="$4"; SUP_SRC="$5"
PIVOT_BIN=$(command -v pivot_root 2>/dev/null || echo /usr/sbin/pivot_root)
[ -x "$PIVOT_BIN" ] || PIVOT_BIN=/sbin/pivot_root
mount --make-rprivate /
mount -t tmpfs -o size=16m,nosuid,nodev tmpfs "$ROOTFS"
mkdir -p "$ROOTFS/usr" "$ROOTFS/tmp" "$ROOTFS/proc" "$ROOTFS/dev" "$ROOTFS/.old_root"
mount --bind /usr "$ROOTFS/usr"
mount -o remount,ro,bind "$ROOTFS/usr"
for sysdir in /lib /lib64 /bin; do
  if [ -L "$sysdir" ]; then
    ln -s "$(readlink "$sysdir")" "$ROOTFS$sysdir"
  elif [ -d "$sysdir" ]; then
    mkdir -p "$ROOTFS$sysdir"
    mount --bind "$sysdir" "$ROOTFS$sysdir"
    mount -o remount,ro,bind "$ROOTFS$sysdir"
  fi
done
if [ -n "$PYBASE" ] && [ -d "$PYBASE" ]; then
  mkdir -p "$ROOTFS$PYBASE"
  mount --bind "$PYBASE" "$ROOTFS$PYBASE"
  mount -o remount,ro,bind "$ROOTFS$PYBASE"
fi
if [ -n "$PYVENV" ] && [ -d "$PYVENV" ]; then
  mkdir -p "$ROOTFS/opt/pyvenv"
  mount --bind "$PYVENV" "$ROOTFS/opt/pyvenv"
  mount -o remount,ro,bind "$ROOTFS/opt/pyvenv"
fi
mount -t tmpfs -o size=64m,nosuid,nodev,noexec tmpfs "$ROOTFS/tmp"
mount -t proc proc "$ROOTFS/proc"
for devnode in null zero urandom; do
  : > "$ROOTFS/dev/$devnode"
  mount --bind "/dev/$devnode" "$ROOTFS/dev/$devnode"
done
"$PIVOT_BIN" "$ROOTFS" "$ROOTFS/.old_root"
cd /tmp
umount -l /.old_root
rmdir /.old_root
exec "$PYEXE" -I -c "$SUP_SRC" --pdeath
"""

def _count_current_uid_tasks() -> int:
    try:
        my_uid, total = os.getuid(), 0
        for entry in os.scandir("/proc"):
            if entry.name.isdigit():
                try:
                    if entry.stat().st_uid == my_uid: total += len(os.listdir(f"/proc/{entry.name}/task"))
                except Exception: pass
        return total
    except Exception: return 0

class LinuxNamespaceSandbox:
    _probe_lock = threading.Lock()
    _probe_cache: Optional[Tuple[bool, str]] = None
    def __init__(self, memory_limit_mb: int = 512, max_processes: int = 16) -> None:
        self.memory_limit_mb, self.max_processes, self.unavailable_reason = memory_limit_mb, max_processes, ""
    @staticmethod
    def _resolve_python_paths() -> Tuple[str, str, str]:
        base_prefix, prefix = os.path.realpath(sys.base_prefix), os.path.realpath(sys.prefix)
        exe_name = os.path.basename(sys.executable) or "python3"
        pybase_arg = "" if base_prefix.startswith("/usr/") or base_prefix == "/usr" else base_prefix
        if prefix != base_prefix: return f"/opt/pyvenv/bin/{exe_name}", pybase_arg, prefix
        return os.path.realpath(sys.executable), pybase_arg, ""
    def _run_raw(self, payload: Dict[str, Any], timeout_s: float, max_response_bytes: int, max_nodes: int) -> Dict[str, Any]:
        task = str(payload.get("task", ""))
        cpu_limit = max(2, int(math.ceil(timeout_s)) + 2)
        uid_tasks = _count_current_uid_tasks()
        wp = dict(payload)
        wp.update({"apply_rlimits": True, "apply_seccomp": True, "memory_limit_mb": self.memory_limit_mb, "cpu_limit_s": cpu_limit, "nproc_limit": max(self.max_processes, uid_tasks + self.max_processes)})
        envelope = {"inner_src": INNER_WORKER_CODE, "runner_src": RUNNER_CODE, "payload": wp, "max_response_bytes": max_response_bytes, "timeout_s": timeout_s}
        req_bytes = json.dumps(envelope, allow_nan=False).encode("utf-8")
        max_req = (MAX_TESTGEN_LARGE_BYTES + 262144) if payload.get("allow_large_request") else (MAX_IPC_REQUEST_BYTES + 262144)
        if len(req_bytes) > max_req: return {"status": "protocol_error", "exception_type": "WorkerProtocolError", "error_msg": "Request exceeded size limit", "execution_time": 0.0, "memory_mb": 0.0}
        inner_exe, pybase_arg, pyvenv_arg = self._resolve_python_paths()
        env = build_sanitized_environment()
        env["PATH"] = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
        with tempfile.TemporaryDirectory(prefix="vgcl_ns_") as temp_dir:
            cmd = ["unshare", "--user", "--map-root-user", "--net", "--pid", "--mount", "--fork", "/bin/sh", "-c", _LINUX_STATIC_MOUNT_SH, "_", temp_dir, inner_exe, pybase_arg, pyvenv_arg, SUPERVISOR_CODE]
            proc = subprocess.Popen(cmd, cwd=temp_dir, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, preexec_fn=os.setpgrp, close_fds=True, shell=False)
            def _kill_ns():
                try: os.killpg(proc.pid, signal.SIGKILL)
                except Exception:
                    try: proc.kill()
                    except Exception: pass
            try:
                timed_out, out_b, err_b, rc = _communicate_bounded(proc, req_bytes, timeout_s=timeout_s + 1.5, max_stdout_bytes=max_response_bytes + 4096, kill_fn=_kill_ns)
            except ValueError as err:
                _kill_ns()
                return {"status": "protocol_error", "exception_type": "WorkerProtocolError", "error_msg": str(err), "execution_time": 0.0, "memory_mb": 0.0}
        if timed_out: return {"status": "timeout", "exception_type": "TimeoutError", "error_msg": f"Execution timed out (> {timeout_s:.1f}s wall clock)", "execution_time": timeout_s, "memory_mb": 0.0}
        if out_b:
            try: return _validate_worker_response(out_b, task=task, max_nodes=max_nodes)
            except Exception as err: return {"status": "protocol_error", "exception_type": "WorkerProtocolError", "error_msg": f"Invalid worker IPC: {err}", "execution_time": 0.0, "memory_mb": 0.0}
        err_s = err_b.decode("utf-8", errors="replace").strip()
        raise SandboxInfrastructureError(f"Namespace sandbox produced no supervisor response (code {rc}): {err_s[:300]}")
    def is_available(self) -> bool:
        if not sys.platform.startswith("linux"):
            self.unavailable_reason = f"Linux namespaces unsupported on {sys.platform}"; return False
        if not shutil.which("unshare"):
            self.unavailable_reason = "'unshare' binary not found in PATH"; return False
        with self._probe_lock:
            if LinuxNamespaceSandbox._probe_cache is not None:
                ok, reason = LinuxNamespaceSandbox._probe_cache; self.unavailable_reason = reason; return ok
            try:
                res = self._run_raw({"task": "probe_self_check"}, timeout_s=5.0, max_response_bytes=MAX_IPC_RESPONSE_BYTES, max_nodes=1024)
                p = res.get("probe") if res.get("status") == "ok" else None
                pids = p.get("proc_pids", []) if isinstance(p, dict) else []
                roots = p.get("root_entries", []) if isinstance(p, dict) else []
                if (isinstance(p, dict) and p.get("usr_readonly") is True and p.get("tmp_writable") is True and p.get("proc_1_clean") is True and p.get("net_blocked") is True and p.get("prctl_blocked") is True and p.get("rlimits_applied") is True and p.get("env_clean") is True and pids and len(pids) <= 10 and not any(d in roots for d in ("home", "root", "etc", "var"))):
                    LinuxNamespaceSandbox._probe_cache = (True, ""); return True
                reason = res.get("error_msg") or f"Probe failed isolation check: {res}"
            except Exception as exc:
                reason = f"Linux namespace probe raised {type(exc).__name__}: {exc}"
            LinuxNamespaceSandbox._probe_cache = (False, reason); self.unavailable_reason = reason; return False
    def execute(self, payload: Dict[str, Any], timeout_s: float = 4.0, max_response_bytes: int = MAX_IPC_RESPONSE_BYTES, max_nodes: int = MAX_IPC_NODES) -> Dict[str, Any]:
        if not self.is_available(): raise SandboxUnavailableError(f"LinuxNamespaceSandbox unavailable: {self.unavailable_reason}")
        return self._run_raw(payload, timeout_s=timeout_s, max_response_bytes=max_response_bytes, max_nodes=max_nodes)

def _cli_environment() -> Dict[str, str]:
    # Environment for the TRUSTED container CLI process only. The sandbox interior
    # receives exclusively the image ENV plus explicit -e flags, never this mapping.
    env = build_sanitized_environment()
    env["PATH"] = os.environ.get("PATH", os.defpath)
    if os.name == "nt":
        for k in ("SYSTEMROOT", "SYSTEMDRIVE", "COMSPEC", "PATHEXT", "TEMP", "TMP",
                  "NUMBER_OF_PROCESSORS", "PROCESSOR_ARCHITECTURE"):
            if k in os.environ: env[k] = os.environ[k]
    return env

class ContainerSandbox:
    def __init__(self, runtime: Optional[str] = None, image: Optional[str] = None, memory_limit_mb: int = 512, max_processes: int = 16) -> None:
        self.memory_limit_mb, self.max_processes, self.unavailable_reason = memory_limit_mb, max_processes, ""
        self.image = image or os.environ.get(ENV_SANDBOX_IMAGE, DEFAULT_CONTAINER_IMAGE)
        if not _IMAGE_REF_RE.match(self.image) or self.image.startswith("-") or not _IMAGE_DIGEST_RE.search(self.image):
            raise SandboxInfrastructureError(f"Container image must be pinned by immutable digest (@sha256:<64 hex>): {self.image!r}")
        rt_opt = os.environ.get(ENV_CONTAINER_RUNTIME, "").strip()
        if rt_opt and not _RUNTIME_OPT_RE.match(rt_opt): raise SandboxInfrastructureError(f"Invalid container runtime option: {rt_opt!r}")
        self.container_runtime = rt_opt or None
        self.runtime = runtime if runtime is not None else self._detect_runtime()
    def _detect_runtime(self) -> Optional[str]:
        reasons = []
        for cand in ("docker", "podman"):
            path = shutil.which(cand)
            if not path: continue
            try:
                if subprocess.run([path, "info"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=8.0, shell=False).returncode != 0:
                    reasons.append(f"{cand} daemon unreachable"); continue
                if subprocess.run([path, "image", "inspect", self.image], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5.0, shell=False).returncode != 0:
                    reasons.append(f"{cand} missing cached image '{self.image}'"); continue
                return path
            except Exception as exc: reasons.append(f"{cand} probe failed: {exc}")
        self.unavailable_reason = "; ".join(reasons) or "neither docker nor podman found in PATH"
        return None
    def is_available(self) -> bool: return self.runtime is not None
    def _force_rm(self, cname: str) -> None:
        if self.runtime:
            try: subprocess.run([self.runtime, "rm", "-f", cname], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5.0, shell=False)
            except Exception: pass
    def execute(self, payload: Dict[str, Any], timeout_s: float = 4.0, max_response_bytes: int = MAX_IPC_RESPONSE_BYTES, max_nodes: int = MAX_IPC_NODES) -> Dict[str, Any]:
        if not self.is_available(): raise SandboxUnavailableError(f"Container sandbox unavailable ({self.unavailable_reason}).")
        task, cpu_limit = str(payload.get("task", "")), max(2, int(math.ceil(timeout_s)) + 2)
        wp = dict(payload); wp.update({"apply_rlimits": True, "apply_seccomp": False, "memory_limit_mb": self.memory_limit_mb, "cpu_limit_s": cpu_limit})
        envelope = {"inner_src": INNER_WORKER_CODE, "runner_src": RUNNER_CODE, "payload": wp, "max_response_bytes": max_response_bytes, "timeout_s": timeout_s}
        req_bytes = json.dumps(envelope, allow_nan=False).encode("utf-8")
        max_req = (MAX_TESTGEN_LARGE_BYTES + 262144) if payload.get("allow_large_request") else (MAX_IPC_REQUEST_BYTES + 262144)
        if len(req_bytes) > max_req: return {"status": "protocol_error", "exception_type": "WorkerProtocolError", "error_msg": "Request exceeded size limit", "execution_time": 0.0, "memory_mb": 0.0}
        cname = f"vgcl-sbx-{os.getpid()}-{uuid.uuid4().hex[:12]}"
        cmd = [self.runtime, "run", "--rm", "--init", "--name", cname, *(["--runtime", self.container_runtime] if self.container_runtime else []), "--pull=never", "--network", "none", "--read-only", "--tmpfs", "/tmp:rw,noexec,nosuid,nodev,size=64m", "--cap-drop", "ALL", "--security-opt", "no-new-privileges", "--ipc", "none", "--user", "65534:65534", "--memory", f"{self.memory_limit_mb}m", "--memory-swap", f"{self.memory_limit_mb}m", "--cpus", "1.0", "--pids-limit", str(self.max_processes), "--ulimit", "fsize=1048576:1048576", "--ulimit", "nofile=64:64", "--ulimit", f"cpu={cpu_limit}:{cpu_limit}", "-e", "PYTHONDONTWRITEBYTECODE=1", "-e", "PYTHONUNBUFFERED=1", "-e", "HOME=/tmp", "-e", "TMPDIR=/tmp", "-e", "OPENBLAS_NUM_THREADS=1", "-e", "OMP_NUM_THREADS=1", "-w", "/tmp", "-i", self.image, "python", "-I", "-c", SUPERVISOR_CODE]
        proc = subprocess.Popen(cmd, env=_cli_environment(), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, close_fds=True, shell=False)
        try:
            timed_out, out_b, err_b, rc = _communicate_bounded(proc, req_bytes, timeout_s=timeout_s + 2.0, max_stdout_bytes=max_response_bytes + 4096, kill_fn=lambda: self._force_rm(cname))
        except ValueError as err:
            self._force_rm(cname)
            return {"status": "protocol_error", "exception_type": "WorkerProtocolError", "error_msg": str(err), "execution_time": 0.0, "memory_mb": 0.0}
        finally:
            if proc.poll() is None: self._force_rm(cname)
        if timed_out: return {"status": "timeout", "exception_type": "TimeoutError", "error_msg": f"Execution timed out (> {timeout_s:.1f}s wall clock)", "execution_time": timeout_s, "memory_mb": 0.0}
        if out_b:
            try: return _validate_worker_response(out_b, task=task, max_nodes=max_nodes)
            except Exception as err: return {"status": "protocol_error", "exception_type": "WorkerProtocolError", "error_msg": f"Invalid container IPC: {err}", "execution_time": 0.0, "memory_mb": 0.0}
        err_s = err_b.decode("utf-8", errors="replace").strip()
        raise SandboxInfrastructureError(f"Container sandbox produced no supervisor response (code {rc}): {err_s[:300]}")

def get_active_sandbox():
    mode = os.environ.get(ENV_SANDBOX_BACKEND, "auto").strip().lower()
    if mode == "container":
        b = ContainerSandbox()
        if not b.is_available(): raise SandboxUnavailableError(f"[FAIL-CLOSED] VGCL_SANDBOX_BACKEND='container' unavailable: {b.unavailable_reason}")
        return b
    if mode == "linux_ns":
        b = LinuxNamespaceSandbox()
        if not b.is_available(): raise SandboxUnavailableError(f"[FAIL-CLOSED] VGCL_SANDBOX_BACKEND='linux_ns' unavailable: {b.unavailable_reason}")
        return b
    if mode == "auto":
        cb = ContainerSandbox()
        if cb.is_available(): return cb
        if sys.platform.startswith("linux"):
            nsb = LinuxNamespaceSandbox()
            if nsb.is_available(): return nsb
            raise SandboxUnavailableError(f"[FAIL-CLOSED] No compliant isolation backend available. ContainerSandbox: {cb.unavailable_reason}; LinuxNamespaceSandbox: {nsb.unavailable_reason}.")
        raise SandboxUnavailableError(f"[FAIL-CLOSED] No compliant isolation backend on '{sys.platform}' ({cb.unavailable_reason}). Non-Linux hosts require Docker/Podman with cached image '{cb.image}'.")
    raise SandboxUnavailableError(f"[FAIL-CLOSED] Unsupported {ENV_SANDBOX_BACKEND}={mode!r}. Valid: 'auto', 'container', 'linux_ns'.")

def _split_monolithic_code(full_code: str) -> Tuple[str, str, str]:
    tree = ast.parse(full_code)
    targets = {"solve", "solve_brute", "generate_test_cases"}
    def _slice(excl: set) -> str:
        return ast.unparse(ast.Module(body=[n for n in tree.body if not (isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in excl)], type_ignores=[]))
    return _slice(targets - {"solve"}), _slice(targets - {"solve_brute"}), _slice(targets - {"generate_test_cases"})

def _run_test_gen_context(sandbox: Any, test_gen_code: str, seed: int, scale: str, max_cases: int, timeout_s: float, max_small_dimension: Optional[int] = None, max_response_bytes: int = MAX_IPC_RESPONSE_BYTES, max_nodes: int = MAX_IPC_NODES) -> Tuple[bool, List[Any], str, str]:
    res = sandbox.execute({"task": "test_gen", "code": test_gen_code, "seed": seed, "scale": scale, "max_cases": max_cases, "max_small_dimension": max_small_dimension}, timeout_s=timeout_s, max_response_bytes=max_response_bytes, max_nodes=max_nodes)
    if res.get("status") != "ok": return False, [], str(res.get("exception_type", "ExecutionError")), str(res.get("error_msg", "test_gen failed"))
    return True, list(res.get("cases", [])), "", ""

def _run_fn_context(sandbox: Any, code: str, fn_name: str, cases: List[Any], timeout_s: float, phase_b_mode: bool = False, consecutive_cases: Optional[List[Any]] = None, allow_large_request: bool = False) -> Dict[str, Any]:
    return sandbox.execute({"task": "run_fn", "code": code, "fn_name": fn_name, "cases": _encode_ipc_value(cases), "phase_b_mode": phase_b_mode, "consecutive_cases": _encode_ipc_value(consecutive_cases) if consecutive_cases else [], "allow_large_request": allow_large_request}, timeout_s=timeout_s)

def execute_candidate_isolated(solve_code: str, test_gen_code: str, solve_brute_code: str, seed: int, timeout_s: float = 4.0, scale: str = "small", max_cases: int = 10) -> Tuple[bool, str, str, float, float]:
    sandbox = get_active_sandbox()
    ok_tg, cases, exc_tg, msg_tg = _run_test_gen_context(sandbox, test_gen_code, seed=seed, scale=scale, max_cases=max_cases, timeout_s=timeout_s)
    if not ok_tg: return False, exc_tg, msg_tg, 0.0, 0.0
    opt_res = _run_fn_context(sandbox, solve_code, "solve", cases, timeout_s=timeout_s)
    opt_time, opt_mem, opt_st = float(opt_res.get("execution_time", 0.0)), float(opt_res.get("memory_mb", 0.0)), opt_res.get("status")
    if opt_st in ("syntax_error", "missing_function", "timeout", "protocol_error", "crash", "unsupported_type"):
        return False, str(opt_res.get("exception_type", "")), str(opt_res.get("error_msg", "")), opt_time, opt_mem
    if not (solve_brute_code and solve_brute_code.strip()):
        if opt_st == "exception": return False, str(opt_res.get("exception_type", "")), str(opt_res.get("error_msg", "")), opt_time, opt_mem
        return True, "", "", opt_time, opt_mem
    opt_fail_idx = int(opt_res["fail_idx"]) if opt_st == "exception" and "fail_idx" in opt_res else len(cases)
    brute_cases = cases[: max(1, opt_fail_idx)] if opt_fail_idx > 0 else []
    brute_res = _run_fn_context(sandbox, solve_brute_code, "solve_brute", brute_cases, timeout_s=timeout_s) if brute_cases else {"status": "ok", "outputs": [], "memory_mb": 0.0}
    b_mem = max(opt_mem, float(brute_res.get("memory_mb", 0.0)))
    if brute_res.get("status") in ("syntax_error", "missing_function", "timeout", "protocol_error", "crash", "unsupported_type"):
        return False, str(brute_res.get("exception_type", "")), str(brute_res.get("error_msg", "")), opt_time, b_mem
    opt_out, brute_out = list(opt_res.get("outputs", [])), list(brute_res.get("outputs", []))
    brute_fail_idx = int(brute_res["fail_idx"]) if brute_res.get("status") == "exception" and "fail_idx" in brute_res else len(brute_cases)
    for idx in range(len(cases)):
        if idx >= opt_fail_idx: return False, str(opt_res.get("exception_type", "")), str(opt_res.get("error_msg", "")), opt_time, b_mem
        if idx >= brute_fail_idx: return False, str(brute_res.get("exception_type", "")), str(brute_res.get("error_msg", "")), opt_time, b_mem
        if not are_results_equal(opt_out[idx], brute_out[idx]):
            return False, "MismatchError", f"Phase A Mismatch: opt={_truncate_repr(opt_out[idx])} brute={_truncate_repr(brute_out[idx])}", opt_time, b_mem
    return True, "", "", opt_time, b_mem

def verify_oracle_phase_a_isolated(full_code: str, seed: int, solve_opt: str = "", solve_brute: str = "", test_gen: str = "", timeout_s: float = 4.0) -> Dict[str, Any]:
    sandbox = get_active_sandbox()
    if solve_opt.strip() and solve_brute.strip() and test_gen.strip(): s_code, b_code, tg_code = solve_opt, solve_brute, test_gen
    else:
        try: s_code, b_code, tg_code = _split_monolithic_code(full_code)
        except SyntaxError as e:
            sp = derive_phase_a_scale_policy(solve_brute or full_code)
            return {"passed": False, "reason": f"Phase A SyntaxError: {e}", "failing_input": None, "expected_output": None, "actual_output": None, "pass_count": 0, "phase_a_scale": sp, "phase_a_scale_cap": int(sp.get("maximum_dimension", 8)), "phase_a_generated_case_count": 0, "phase_a_admissible_case_count": 0, "phase_a_scale_policy": sp.get("policy", "unknown")}
    sp = derive_phase_a_scale_policy(b_code or full_code); cap = int(sp.get("maximum_dimension", 8))
    def _fail(reason: str, pass_count: int = 0, gen_cnt: int = 0, adm_cnt: int = 0, fi: Optional[str] = None, exp: Optional[str] = None, act: Optional[str] = None) -> Dict[str, Any]:
        return {"passed": False, "reason": reason, "failing_input": fi, "expected_output": exp, "actual_output": act, "pass_count": pass_count, "phase_a_scale": sp, "phase_a_scale_cap": cap, "phase_a_generated_case_count": gen_cnt, "phase_a_admissible_case_count": adm_cnt, "phase_a_scale_policy": sp.get("policy", "unknown")}
    ok_tg, raw_cases, exc_tg, msg_tg = _run_test_gen_context(sandbox, tg_code, seed=seed, scale="small", max_cases=20, max_small_dimension=cap, timeout_s=timeout_s)
    if not ok_tg: return _fail("Phase A Timeout (> 4s wall clock)" if exc_tg == "TimeoutError" else f"Phase A Exception: {exc_tg}: {msg_tg}")
    gen_cnt = len(raw_cases); test_cases = [a for a in raw_cases if case_within_phase_a_policy(a, sp)]; adm_cnt = len(test_cases)
    if not test_cases: return _fail(f"Phase A Scale Policy Rejected: no generated small cases within brute-force dimension cap ({cap})", gen_cnt=gen_cnt, adm_cnt=0)
    opt_res = _run_fn_context(sandbox, s_code, "solve", test_cases, timeout_s=timeout_s); opt_st = opt_res.get("status")
    if opt_st == "timeout": return _fail("Phase A Timeout (> 4s wall clock)", gen_cnt=gen_cnt, adm_cnt=adm_cnt)
    if opt_st in ("syntax_error", "missing_function", "protocol_error", "crash", "unsupported_type"): return _fail(f"Phase A Exception: {opt_res.get('exception_type')}: {opt_res.get('error_msg')}", gen_cnt=gen_cnt, adm_cnt=adm_cnt)
    opt_fail = int(opt_res["fail_idx"]) if opt_st == "exception" and "fail_idx" in opt_res else len(test_cases)
    b_cases = test_cases[: max(1, opt_fail)] if opt_fail > 0 else []
    b_res = _run_fn_context(sandbox, b_code, "solve_brute", b_cases, timeout_s=timeout_s) if b_cases else {"status": "ok", "outputs": []}; b_st = b_res.get("status")
    if b_st == "timeout": return _fail("Phase A Timeout (> 4s wall clock)", gen_cnt=gen_cnt, adm_cnt=adm_cnt)
    if b_st in ("syntax_error", "missing_function", "protocol_error", "crash", "unsupported_type"): return _fail(f"Phase A Exception: {b_res.get('exception_type')}: {b_res.get('error_msg')}", gen_cnt=gen_cnt, adm_cnt=adm_cnt)
    if float(opt_res.get("memory_mb", 0.0)) > 512.0 or float(b_res.get("memory_mb", 0.0)) > 512.0: return _fail("Phase A Memory Limit Exceeded (> 512MB)", gen_cnt=gen_cnt, adm_cnt=adm_cnt)
    opt_out, b_out = list(opt_res.get("outputs", [])), list(b_res.get("outputs", []))
    b_fail = int(b_res["fail_idx"]) if b_st == "exception" and "fail_idx" in b_res else len(b_cases)
    pass_cnt = 0
    for idx, args in enumerate(test_cases):
        pa = args if isinstance(args, (list, tuple)) else (args,)
        if idx >= opt_fail: return _fail(f"Phase A Exception: {opt_res.get('exception_type')}: {opt_res.get('error_msg')}", pass_cnt, gen_cnt, adm_cnt, _truncate_repr(pa))
        if idx >= b_fail: return _fail(f"Phase A Exception: {b_res.get('exception_type')}: {b_res.get('error_msg')}", pass_cnt, gen_cnt, adm_cnt, _truncate_repr(pa))
        if not are_results_equal(opt_out[idx], b_out[idx]): return _fail(f"Phase A Output Mismatch on test #{idx + 1}", pass_cnt, gen_cnt, adm_cnt, _truncate_repr(pa), _truncate_repr(b_out[idx]), _truncate_repr(opt_out[idx]))
        pass_cnt += 1
    return {"passed": True, "reason": "Phase A Passed", "pass_count": pass_cnt, "phase_a_scale": sp, "phase_a_scale_cap": cap, "phase_a_generated_case_count": gen_cnt, "phase_a_admissible_case_count": adm_cnt, "phase_a_scale_policy": sp.get("policy", "unknown")}

def verify_oracle_phase_b_isolated(full_code: str, seed: int, solve_opt: str = "", test_gen: str = "", spec: str = "", timeout_s: float = 12.0) -> Dict[str, Any]:
    sandbox = get_active_sandbox(); constraints = extract_declared_constraints(spec); max_n = constraints.get("max_n", 100000)
    def _bfail(reason: str, scales: List[int], rts: List[float], cr_reason: str, fi: Optional[str] = None, act: Optional[str] = None) -> Dict[str, Any]:
        return {"passed": False, "reason": reason, "failing_input": fi, "actual_output": act, "declared_constraint_summary": constraints.get("summary", ""), "large_scale_parameter": max_n, "phase_b_test_scales": scales, "observed_runtime_per_scale": rts, "constraint_regime_status": "INCOMPATIBLE", "constraint_regime_reason": cr_reason}
    if solve_opt.strip() and test_gen.strip(): s_code, tg_code = solve_opt, test_gen
    else:
        try: s_code, _, tg_code = _split_monolithic_code(full_code)
        except SyntaxError as e: return _bfail(f"Phase B Exception: SyntaxError: {e}", [], [], "SyntaxError in Phase B")
    ok_tg, large_cases, exc_tg, msg_tg = _run_test_gen_context(sandbox, tg_code, seed=seed, scale="large", max_cases=5, timeout_s=timeout_s, max_response_bytes=MAX_TESTGEN_LARGE_BYTES, max_nodes=MAX_IPC_LARGE_NODES)
    if not ok_tg: return _bfail(f"Phase B Exception: {exc_tg}: {msg_tg}", [], [], f"Exception during test_gen: {exc_tg}")
    obs_scales = [_max_dimension(a if isinstance(a, (list, tuple)) else (a,)) for a in large_cases]
    opt_res = _run_fn_context(sandbox, s_code, "solve", large_cases, timeout_s=timeout_s, phase_b_mode=True, allow_large_request=True)
    st, rts = opt_res.get("status"), list(opt_res.get("per_case_times", []))
    used_scales = obs_scales[: len(rts)]
    if st == "timeout": return _bfail("Phase B Timeout (> 12s wall clock)", used_scales, rts, "Phase B process exceeded 12s wall clock limit")
    if st == "timeout_case":
        fi = int(opt_res.get("fail_idx", 0)); dim = obs_scales[fi] if fi < len(obs_scales) else 0; dt = rts[-1] if rts else 2.0
        fa = large_cases[fi] if fi < len(large_cases) else None
        return _bfail(f"Phase B Time Bound Violation: Large test #{fi + 1} (dim={dim}) took {dt:.4f}s (> 2.0s)", used_scales, rts, f"Runtime {dt:.4f}s exceeded 2.0s timeout at scale {dim}", _truncate_repr(fa) if fa is not None else None, opt_res.get("actual_preview"))
    if st != "ok":
        fi = int(opt_res.get("fail_idx", 0)); fa = large_cases[fi] if fi < len(large_cases) else None; exc = opt_res.get("exception_type", "ExecutionError")
        return _bfail(f"Phase B Exception: {exc}: {opt_res.get('error_msg', '')}", used_scales, rts, f"Exception during execution: {exc}", _truncate_repr(fa) if fa is not None else None)
    if float(opt_res.get("memory_mb", 0.0)) > 512.0: return _bfail("Phase B Memory Limit Exceeded (> 512MB)", used_scales, rts, "Memory limit exceeded")
    return {"passed": True, "reason": "Phase B Passed", "declared_constraint_summary": constraints.get("summary", ""), "large_scale_parameter": max_n, "phase_b_test_scales": used_scales, "observed_runtime_per_scale": rts, "constraint_regime_status": "COMPATIBLE", "constraint_regime_reason": "Observed runtime and memory scaling compatible with declared regime"}

def audit_entry_isolated(code: str, seed: int, timeout_s: float = 4.0) -> List[str]:
    sandbox = get_active_sandbox()
    try: s_code, b_code, tg_code = _split_monolithic_code(code)
    except SyntaxError as ex: return [f"CODE_EXEC_ERROR: SyntaxError: {ex}"]
    flags: List[str] = []
    ok_s, small_cases, exc_s, msg_s = _run_test_gen_context(sandbox, tg_code, seed=seed or 42, scale="small", max_cases=20, timeout_s=timeout_s)
    if not ok_s: return ["MISSING_GEN_FN: generate_test_cases() not found after exec"] if exc_s == "MissingFunction" else [f"TESTGEN_EXEC_ERROR(small): {exc_s}: {msg_s}"]
    if not small_cases: return ["TESTGEN_EMPTY: generate_test_cases returns no cases for scale=small"]
    all_int = all(isinstance(args, (list, tuple)) and all(isinstance(a, int) and not isinstance(a, bool) for a in (args if isinstance(args, (list, tuple)) else (args,))) for args in small_cases)
    if all_int: flags.append("TESTGEN_INT_ARGS: all test case args are bare integers (no real inputs)")
    check_cases = small_cases[:10]
    opt_res = _run_fn_context(sandbox, s_code, "solve", check_cases, timeout_s=timeout_s)
    if opt_res.get("status") == "missing_function": flags.append("MISSING_SOLVE_FN: solve() not found after exec")
    elif opt_res.get("status") != "ok": flags.append(f"SOLVE_EXEC_ERROR: {opt_res.get('exception_type')}: {opt_res.get('error_msg')}")
    opt_out = list(opt_res.get("outputs", [])); diff_fail = False
    if not all_int and opt_res.get("status") == "ok":
        b_res = _run_fn_context(sandbox, b_code, "solve_brute", check_cases, timeout_s=timeout_s)
        if b_res.get("status") == "missing_function": flags.append("MISSING_BRUTE_FN: solve_brute() not found after exec")
        elif b_res.get("status") != "ok": flags.append(f"BRUTE_EXEC_ERROR: {b_res.get('exception_type')}: {b_res.get('error_msg')}")
        else:
            for r_opt, r_b in zip(opt_out, list(b_res.get("outputs", []))):
                if not are_results_equal(r_opt, r_b):
                    flags.append(f"DIFFERENTIAL_FAIL: solve={_truncate_repr(r_opt)} vs solve_brute={_truncate_repr(r_b)}"); diff_fail = True; break
    if opt_out and not diff_fail and opt_res.get("status") == "ok" and len(set(repr(r) for r in opt_out)) == 1 and len(opt_out) > 1:
        flags.append(f"CONSTANT_OUTPUT: solve returns {_truncate_repr(opt_out[0])} on every test case")
    ok_l, large_cases, exc_l, msg_l = _run_test_gen_context(sandbox, tg_code, seed=seed or 42, scale="large", max_cases=20, timeout_s=timeout_s, max_response_bytes=MAX_TESTGEN_LARGE_BYTES, max_nodes=MAX_IPC_LARGE_NODES)
    if not ok_l: flags.append(f"TESTGEN_EXEC_ERROR(large): {exc_l}: {msg_l}")
    elif not large_cases: flags.append("LARGE_EMPTY: generate_test_cases returns nothing for scale=large")
    else:
        def _isz(ca): return max((len(a) if isinstance(a, (list, tuple, str)) else (a if isinstance(a, int) and not isinstance(a, bool) else 0) for a in ca), default=0)
        ss = max((_isz(tuple(a) if isinstance(a, (list, tuple)) else (a,)) for a in small_cases), default=0)
        ls = max((_isz(tuple(a) if isinstance(a, (list, tuple)) else (a,)) for a in large_cases), default=0)
        if ls <= ss: flags.append(f"LARGE_NOT_BIGGER: large inputs (size={ls}) are not bigger than small inputs (size={ss})")
    return flags

def verify_legacy_phase_isolated(phase: str, code_str: str, seed: int, timeout_s: float = 4.0) -> Tuple[bool, str, Optional[str]]:
    sandbox = get_active_sandbox()
    try: s_code, b_code, tg_code = _split_monolithic_code(code_str)
    except SyntaxError as e: return False, f"Phase {'A' if phase == 'a' else 'B/C'} Runtime Exception: SyntaxError: {e}", "N/A"
    if phase == "a":
        ok_tg, tcs, exc_tg, msg_tg = _run_test_gen_context(sandbox, tg_code, seed=seed, scale="small", max_cases=20, timeout_s=timeout_s)
        if not ok_tg: return False, f"Phase A Runtime Exception: {exc_tg}: {msg_tg}", "N/A"
        def _is_lg(a): return (isinstance(a, (list, tuple, str, dict, set)) and len(a) >= 10000) or (isinstance(a, (int, float)) and not isinstance(a, bool) and a > 2000000000)
        opt_res = _run_fn_context(sandbox, s_code, "solve", tcs, timeout_s=timeout_s)
        if opt_res.get("status") not in ("ok", "exception"): return False, f"Phase A Runtime Exception: {opt_res.get('exception_type')}: {opt_res.get('error_msg')}", "N/A"
        opt_fail = int(opt_res["fail_idx"]) if opt_res.get("status") == "exception" else len(tcs)
        non_lg = [i for i, args in enumerate(tcs[:opt_fail]) if not any(_is_lg(a) for a in (args if isinstance(args, (list, tuple)) else (args,)))]
        b_in = [tcs[i] for i in non_lg]
        b_res = _run_fn_context(sandbox, b_code, "solve_brute", b_in, timeout_s=timeout_s) if b_in else {"status": "ok", "outputs": []}
        if b_res.get("status") not in ("ok", "exception"): return False, f"Phase A Runtime Exception: {b_res.get('exception_type')}: {b_res.get('error_msg')}", "N/A"
        b_out = list(b_res.get("outputs", [])); b_fail = int(b_res["fail_idx"]) if b_res.get("status") == "exception" else len(b_in)
        b_map = {oi: p for p, oi in enumerate(non_lg)}; opt_out = list(opt_res.get("outputs", []))
        for idx, args in enumerate(tcs):
            pa = args if isinstance(args, (list, tuple)) else (args,)
            if idx >= opt_fail: return False, f"Phase A Runtime Exception: {opt_res.get('exception_type')}: {opt_res.get('error_msg')}", _truncate_repr(pa)
            if idx in b_map:
                bp = b_map[idx]
                if bp >= b_fail: return False, f"Phase A Runtime Exception: {b_res.get('exception_type')}: {b_res.get('error_msg')}", _truncate_repr(pa)
                if not are_results_equal(opt_out[idx], b_out[bp]):
                    return False, f"Phase A Mismatch on small test #{idx + 1}: Optimal={_truncate_repr(opt_out[idx])} vs Brute={_truncate_repr(b_out[bp])}", _truncate_repr(pa)
        return True, "Phase A Passed", None
    ok_sm, sm_c, exc_sm, msg_sm = _run_test_gen_context(sandbox, tg_code, seed=seed, scale="small", max_cases=3, timeout_s=timeout_s)
    if not ok_sm: return False, f"Phase B/C Exception: {exc_sm}: {msg_sm}", "N/A"
    ok_lg, lg_c, exc_lg, msg_lg = _run_test_gen_context(sandbox, tg_code, seed=seed, scale="large", max_cases=5, timeout_s=timeout_s, max_response_bytes=MAX_TESTGEN_LARGE_BYTES, max_nodes=MAX_IPC_LARGE_NODES)
    if not ok_lg: return False, f"Phase B/C Exception: {exc_lg}: {msg_lg}", "N/A"
    opt_res = _run_fn_context(sandbox, s_code, "solve", lg_c, timeout_s=timeout_s, phase_b_mode=True, consecutive_cases=sm_c[:3], allow_large_request=True)
    st = opt_res.get("status")
    if st == "timeout": return False, f"Phase C Timeout: Execution timed out (> {timeout_s:.1f}s)", "N/A"
    if st == "timeout_case":
        fi = int(opt_res.get("fail_idx", 0)); dt = opt_res.get("per_case_times", [2.0])[-1]; fa = lg_c[fi] if fi < len(lg_c) else None
        return False, f"Phase C Timeout: Large scale test #{fi + 1} took {dt:.4f}s (> 2.0s)", _truncate_repr(fa)
    if st != "ok": return False, f"Phase B/C Exception: {opt_res.get('exception_type')}: {opt_res.get('error_msg')}", "N/A"
    for c_idx, pair in enumerate(opt_res.get("consecutive_pairs", [])):
        if not are_results_equal(pair[0], pair[1]):
            ca = sm_c[c_idx] if c_idx < len(sm_c) else None
            return False, "Phase B State Mutation Failure: consecutive calls produced different results.", _truncate_repr(ca)
    return True, "Phase B & C Passed", None
