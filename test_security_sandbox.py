"""test_security_sandbox.py

Security and Sandboxing Regression Test Suite for VGCL Pipeline.

Validates that:
  1. Candidate code, reference solutions, and test generators execute strictly
     inside OS-isolated sandbox backends (OCI container / Linux namespaces+seccomp).
  2. Untrusted code cannot access parent process memory, credentials, inherited
     descriptors, or sensitive environment variables.
  3. Host filesystem and network (IPv4/IPv6/TCP/UDP/DNS/loopback) are protected
     by real kernel enforcement, not Python-level monkey-patching.
  4. Timeouts, process-tree cleanup, memory/file-size limits, and output flooding
     are enforced.
  5. The system fails closed if no compliant sandbox backend is available;
     infrastructure failures raise SandboxInfrastructureError instead of being
     mislabeled as candidate failures.
  6. IPC uses strict, lossless tagged JSON (no pickle) with framing, size, depth,
     node-budget, duplicate-key, and non-finite rejection; workers can never
     assert 'passed'.
  7. Verification contracts, failure classification, and model supply-chain
     configs are preserved.

Integration tests requiring a live sandbox backend are skipped (with an explicit
reason) when no compliant backend exists on the host; pure codec/policy/contract
tests always run.
"""

import ast
import errno
import inspect
import json
import math
import os
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import time
import unittest

import isolated_executor as ie
from isolated_executor import (
    DEFAULT_CONTAINER_IMAGE,
    MAX_IPC_DEPTH,
    SandboxInfrastructureError,
    SandboxUnavailableError,
    _decode_ipc_value,
    _encode_ipc_value,
    _validate_worker_response,
    are_results_equal,
    audit_entry_isolated,
    build_sanitized_environment,
    execute_candidate_isolated,
    get_active_sandbox,
)

CANARY_VAR = "VGCL_TEST_SECRET_CANARY"
CANARY_VAL = "SUPER_SECRET_TOKEN_DO_NOT_LEAK_998877"

SIMPLE_TGEN = (
    "import random\n"
    "def generate_test_cases(seed, scale):\n"
    "    random.seed(seed)\n"
    "    return [([random.randint(-10, 10) for _ in range(5)],)]\n"
)


def _run_solve(sandbox, code, cases=None, timeout_s=8.0):
    """Single run_fn invocation against the active backend (one sandbox startup)."""
    payload = {
        "task": "run_fn",
        "code": code,
        "fn_name": "solve",
        "cases": _encode_ipc_value(cases if cases is not None else [(1,)]),
        "phase_b_mode": False,
        "consecutive_cases": [],
    }
    return sandbox.execute(payload, timeout_s=timeout_s)


class TestIPCCodec(unittest.TestCase):
    """Lossless tagged-JSON IPC codec: round trips and hostile-input rejection."""

    def _roundtrip(self, val):
        wire = json.dumps(_encode_ipc_value(val), allow_nan=False)
        return _decode_ipc_value(json.loads(wire))

    def test_01_scalar_roundtrips(self):
        for val in (None, True, False, 0, -1, 2**200, "", "héllo\x00世界", 3.25, -0.0):
            out = self._roundtrip(val)
            self.assertEqual(out, val)
            self.assertIs(type(out), type(val))

    def test_02_special_float_roundtrips(self):
        for val in (float("nan"), float("inf"), float("-inf")):
            enc = _encode_ipc_value(val)
            self.assertEqual(enc["__vgcl_type__"], "float")
            out = self._roundtrip(val)
            if math.isnan(val):
                self.assertTrue(math.isnan(out))
            else:
                self.assertEqual(out, val)

    def test_03_bytes_roundtrip(self):
        for val in (b"", bytes(range(256)), b"\x00\xff" * 1000):
            out = self._roundtrip(val)
            self.assertEqual(out, val)
            self.assertIsInstance(out, bytes)

    def test_04_container_roundtrips(self):
        vals = [
            [1, "a", None, [2.5]],
            (1, "a", None, (2.5,)),
            {1, 2, 3},
            frozenset({("x", 1), (None, 2)}),
            [(1,), {2}, frozenset({3})],
        ]
        for val in vals:
            out = self._roundtrip(val)
            self.assertEqual(out, val)
            self.assertIs(type(out), type(val))

    def test_05_dict_nonstring_keys_roundtrip(self):
        val = {1: "int", (2, 3): "tuple", None: "none", frozenset({4}): "fset", 5.5: "float", "k": b"bytes"}
        out = self._roundtrip(val)
        self.assertEqual(out, val)
        self.assertIs(type(out), dict)

    def test_06_tag_collision_dict_roundtrip(self):
        val = {"__vgcl_type__": "dict", "items": "not-a-real-payload"}
        out = self._roundtrip(val)
        self.assertEqual(out, val)

    def test_07_numpy_scalar_roundtrip(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy not installed")
        for val in (np.int64(7), np.float64(1.5), np.bool_(True), np.float32(0.5)):
            out = self._roundtrip(val)
            self.assertEqual(out, val.item())

    def test_08_no_value_truncation(self):
        big_str = "x" * 100_000
        big_int = 10**5000
        self.assertEqual(self._roundtrip(big_str), big_str)
        self.assertEqual(self._roundtrip(big_int), big_int)

    def test_09_reject_duplicate_dict_keys(self):
        enc = {"__vgcl_type__": "dict", "items": [[1, "a"], [1, "b"]]}
        with self.assertRaises(ValueError):
            _decode_ipc_value(enc)

    def test_10_reject_unhashable_dict_key(self):
        enc = {"__vgcl_type__": "dict", "items": [[[1, 2], "v"]]}
        with self.assertRaises(ValueError):
            _decode_ipc_value(enc)

    def test_11_reject_unknown_tag(self):
        with self.assertRaises(ValueError):
            _decode_ipc_value({"__vgcl_type__": "pickle", "data": "x"})

    def test_12_reject_malformed_base64(self):
        for bad in ("!!!not-base64!!!", "QQ=", "a"):
            with self.assertRaises(Exception):
                _decode_ipc_value({"__vgcl_type__": "bytes", "b64": bad})

    def test_13_reject_excessive_nesting(self):
        val = 0
        for _ in range(MAX_IPC_DEPTH + 5):
            val = [val]
        with self.assertRaises(ValueError):
            _encode_ipc_value(val)

    def test_14_reject_cycle(self):
        val = []
        val.append(val)
        with self.assertRaises(ValueError):
            _encode_ipc_value(val)

    def test_15_reject_node_budget(self):
        enc = _encode_ipc_value(list(range(100)))
        with self.assertRaises(ValueError):
            _decode_ipc_value(enc, budget=[10])

    def test_16_reject_nonfinite_raw_float(self):
        with self.assertRaises(ValueError):
            _decode_ipc_value(float("inf"))
        with self.assertRaises(ValueError):
            json.loads('{"x": NaN}', parse_constant=ie._reject_nonfinite_json)
        with self.assertRaises(ValueError):
            json.loads('{"x": Infinity}', parse_constant=ie._reject_nonfinite_json)

    def test_17_reject_unsupported_type(self):
        with self.assertRaises(TypeError):
            _encode_ipc_value(object())


class TestResultsEqual(unittest.TestCase):
    """Differential comparison semantics preserved (trusted-parent verdicts)."""

    def test_nan_equality(self):
        self.assertTrue(are_results_equal(float("nan"), float("nan")))

    def test_bool_int_distinction(self):
        self.assertFalse(are_results_equal(True, 1))
        self.assertFalse(are_results_equal(False, 0))

    def test_list_tuple_coercion(self):
        self.assertTrue(are_results_equal([1, 2], (1, 2)))
        self.assertFalse(are_results_equal([1, 2], (1, 2, 3)))

    def test_float_tolerance(self):
        self.assertTrue(are_results_equal(1.0, 1.0 + 1e-9))
        self.assertFalse(are_results_equal(1.0, 1.1))

    def test_nested_dict(self):
        self.assertTrue(are_results_equal({"a": [1, (2,)]}, {"a": (1, [2])}))
        self.assertFalse(are_results_equal({"a": 1}, {"a": 2}))

    def test_numpy_unwrap(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy not installed")
        self.assertTrue(are_results_equal(np.int64(3), 3))
        self.assertFalse(are_results_equal(np.bool_(True), 1))


class TestWorkerResponseValidation(unittest.TestCase):
    """Trusted-parent framing/schema validation of worker responses."""

    def _ok_run_fn(self):
        return {"status": "ok", "outputs": [1], "execution_time": 0.1, "memory_mb": 10.0}

    def test_valid_minimal_response(self):
        data = _validate_worker_response(json.dumps(self._ok_run_fn()).encode(), task="run_fn")
        self.assertEqual(data["outputs"], [1])

    def test_reject_passed_assertion(self):
        resp = self._ok_run_fn()
        resp["passed"] = True
        with self.assertRaises(ValueError):
            _validate_worker_response(json.dumps(resp).encode(), task="run_fn")

    def test_reject_unexpected_keys(self):
        resp = self._ok_run_fn()
        resp["verdict"] = "APPROVED"
        with self.assertRaises(ValueError):
            _validate_worker_response(json.dumps(resp).encode(), task="run_fn")

    def test_reject_unknown_status(self):
        resp = self._ok_run_fn()
        resp["status"] = "definitely_ok"
        with self.assertRaises(ValueError):
            _validate_worker_response(json.dumps(resp).encode(), task="run_fn")

    def test_reject_ok_without_required_fields(self):
        for resp in ({"status": "ok"}, {"status": "ok", "outputs": [1]},
                     {"status": "ok", "outputs": [1], "execution_time": 0.1}):
            with self.assertRaises(ValueError):
                _validate_worker_response(json.dumps(resp).encode(), task="run_fn")

    def test_reject_nonfinite_json(self):
        with self.assertRaises(ValueError):
            _validate_worker_response(b'{"status":"ok","outputs":[1],"execution_time":NaN,"memory_mb":1.0}', task="run_fn")

    def test_reject_multi_frame_and_trailing(self):
        frame = json.dumps(self._ok_run_fn()).encode()
        for raw in (frame + b"\n" + frame, frame + b"garbage", b"not json"):
            with self.assertRaises(ValueError):
                _validate_worker_response(raw, task="run_fn")

    def test_reject_invalid_numerics(self):
        for bad_time in (-1.0, True, "0.1"):
            resp = self._ok_run_fn()
            resp["execution_time"] = bad_time
            with self.assertRaises(ValueError):
                _validate_worker_response(json.dumps(resp).encode(), task="run_fn")

    def test_reject_unknown_task(self):
        with self.assertRaises(ValueError):
            _validate_worker_response(json.dumps(self._ok_run_fn()).encode(), task="root_shell")


class TestSecuritySandbox(unittest.TestCase):
    """Fail-closed backend policy, supply-chain, and routing contract tests (always run)."""

    def test_19_fail_closed_deterministic_backend_policy(self):
        old = os.environ.get("VGCL_SANDBOX_BACKEND")
        try:
            for bad in ("process", "job_object", "in_process", "non_existent_sandbox_backend_xyz"):
                os.environ["VGCL_SANDBOX_BACKEND"] = bad
                with self.assertRaises(SandboxUnavailableError, msg=f"backend={bad}"):
                    get_active_sandbox()
            if os.name == "nt":
                os.environ["VGCL_SANDBOX_BACKEND"] = "linux_ns"
                with self.assertRaises(SandboxUnavailableError):
                    get_active_sandbox()
        finally:
            if old is None:
                os.environ.pop("VGCL_SANDBOX_BACKEND", None)
            else:
                os.environ["VGCL_SANDBOX_BACKEND"] = old

    def test_unpinned_container_images_rejected(self):
        for img in ("python:3.12-slim", "python:latest", "python", "evil@sha256:zzzz"):
            with self.assertRaises(SandboxInfrastructureError, msg=img):
                ie.ContainerSandbox(runtime="docker", image=img)

    def test_digest_pinned_image_accepted(self):
        sb = ie.ContainerSandbox(runtime="docker", image=DEFAULT_CONTAINER_IMAGE)
        self.assertIn("@sha256:", sb.image)
        self.assertEqual(len(sb.image.split("@sha256:")[1]), 64)

    def test_no_process_only_fallback_exists(self):
        for name in ("LinuxProcessSandbox", "WindowsJobObjectSandbox", "block_network", "restore_network"):
            self.assertFalse(hasattr(ie, name), f"legacy process-only symbol still present: {name}")
        src = inspect.getsource(ie)
        self.assertNotIn("import pickle", src)
        self.assertNotIn("multiprocessing", src)

    def test_sanitized_environment_excludes_host_secrets(self):
        os.environ[CANARY_VAR] = CANARY_VAL
        os.environ["GEMINI_API_KEY"] = "MOCK_GEMINI_KEY_ABC123"
        try:
            env = build_sanitized_environment()
            self.assertNotIn(CANARY_VAR, env)
            self.assertNotIn("GEMINI_API_KEY", env)
            self.assertNotIn(CANARY_VAL, json.dumps(env))
        finally:
            os.environ.pop(CANARY_VAR, None)
            os.environ.pop("GEMINI_API_KEY", None)

    def test_prober_delegates_to_isolated_executor(self):
        from base_model_prober import BaseModelProber
        src = inspect.getsource(BaseModelProber._execute_candidate)
        self.assertIn("execute_candidate_isolated", src)
        self.assertNotIn("exec(", src)

    def test_trust_remote_code_disabled(self):
        import base_model_prober
        for path in (base_model_prober.__file__,
                     os.path.join(os.path.dirname(base_model_prober.__file__), "phase0_calibration", "prepare_calibration_data.py")):
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            self.assertNotIn("trust_remote_code=True", content)
            self.assertIn("trust_remote_code=False", content)

    def test_oracle_routes_through_central_isolation(self):
        import oracle_refactor
        src = inspect.getsource(oracle_refactor)
        self.assertIn("verify_oracle_phase_a_isolated", src)
        self.assertIn("verify_oracle_phase_b_isolated", src)
        self.assertNotIn("pickle", src)
        self.assertNotIn("multiprocessing", src)

    def test_legacy_routes_through_central_isolation(self):
        from legacy import verifier, audit_curriculum
        vsrc = inspect.getsource(verifier)
        self.assertIn("verify_legacy_phase_isolated", vsrc)
        self.assertIn("SandboxInfrastructureError", vsrc)
        asrc = inspect.getsource(audit_curriculum)
        self.assertIn("audit_entry_isolated", asrc)
        self.assertIn("SandboxInfrastructureError", asrc)

    def test_kaggle_probe_preflight_fails_closed(self):
        import kaggle_probe
        src = inspect.getsource(kaggle_probe.run_kaggle_probing)
        self.assertIn("get_active_sandbox()", src)

    def test_ci_workflow_pins_digest(self):
        wf = os.path.join(os.path.dirname(ie.__file__), ".github", "workflows", "security-tests.yml")
        with open(wf, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("docker pull python@sha256:", content)
        self.assertNotIn("docker pull python:3.12-slim", content)

    def test_legacy_ast_blocks_malicious_fixture(self):
        from legacy.verifier import verify_ast_anti_cheat
        malicious = (
            "import os\n"
            "def solve(*args):\n"
            "    os.system('echo HACKED')\n"
            "    return 0\n"
            "def solve_brute(*args): return 0\n"
            "def generate_test_cases(seed, scale): return [(1,)]\n"
        )
        ok, msg = verify_ast_anti_cheat(malicious)
        self.assertFalse(ok)


def _simulate_seccomp_bpf(insns, nr, arch, arg0=0):
    """Independent classic-BPF interpreter for the seccomp filter under test.

    seccomp_data layout: nr@0, arch@4, instruction_pointer@8, args[0]@16.
    Supports exactly the opcodes the policy filter uses; anything else is a
    test failure, not a silent pass.
    """
    prog = [struct.unpack("HBBI", insns[i:i + 8]) for i in range(0, len(insns), 8)]
    data = {0: nr & 0xFFFFFFFF, 4: arch & 0xFFFFFFFF, 8: 0, 16: arg0 & 0xFFFFFFFF}
    a, pc = 0, 0
    for _ in range(len(prog) + 1):
        if pc >= len(prog):
            raise AssertionError(f"BPF fell off end of program at pc={pc}")
        code, jt, jf, k = prog[pc]
        if code == 0x20:  # BPF_LD|BPF_W|BPF_ABS
            a = data.get(k, 0)
            pc += 1
        elif code == 0x15:  # BPF_JMP|BPF_JEQ|BPF_K
            pc += 1 + (jt if a == k else jf)
        elif code == 0x54:  # BPF_ALU|BPF_AND|BPF_K
            a &= k
            pc += 1
        elif code == 0x06:  # BPF_RET|BPF_K
            return k
        else:
            raise AssertionError(f"Unexpected BPF opcode {code:#x} at pc={pc}")
    raise AssertionError("BPF program did not terminate (possible loop)")


class TestSeccompBPFControlFlow(unittest.TestCase):
    """Independent control-flow audit of the real seccomp-BPF filter.

    Extracts build_seccomp_insns from INNER_WORKER_CODE (the exact source shipped
    to the sandbox) and simulates it against the intended policy. Pure Python:
    runs on every host, no Linux required.
    """

    RET_ALLOW = 0x7FFF0000
    RET_EPERM = 0x00050000 | errno.EPERM
    RET_ENOSYS = 0x00050000 | errno.ENOSYS
    NS_MASK = 0x7E020000
    NS_FLAGS = [0x00020000, 0x02000000, 0x04000000, 0x08000000, 0x10000000, 0x20000000, 0x40000000]
    # Intended policy, stated independently of the implementation.
    POLICY = {
        "x86_64": (0xC000003E, 56, 435,
                   [41, 42, 43, 44, 46, 49, 50, 101, 155, 157, 161, 165, 166, 175, 176,
                    246, 248, 249, 250, 272, 288, 298, 307, 308, 310, 311, 313, 321, 323,
                    425, 428, 429, 430, 431, 432, 433, 438, 442]),
        "aarch64": (0xC00000B7, 220, 435,
                    [39, 40, 41, 51, 97, 104, 105, 106, 117, 167, 198, 200, 201, 202, 203,
                     206, 211, 217, 218, 219, 241, 242, 268, 269, 270, 271, 273, 280, 282,
                     425, 428, 429, 430, 431, 432, 433, 438, 442]),
    }

    @classmethod
    def setUpClass(cls):
        tree = ast.parse(ie.INNER_WORKER_CODE)
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "build_seccomp_insns")
        ns = {"struct": struct, "errno": errno}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), "<seccomp_builder>", "exec"), ns)
        cls.build = staticmethod(ns["build_seccomp_insns"])
        cls.filters = {name: b"".join(cls.build(*params)) for name, params in cls.POLICY.items()}

    def _expected(self, params, nr, arch, arg0):
        want_arch, clone_nr, clone3_nr, blocked = params
        if arch != want_arch: return self.RET_EPERM
        if nr in blocked: return self.RET_EPERM
        if nr == clone3_nr: return self.RET_ENOSYS
        if nr == clone_nr and (arg0 & self.NS_MASK): return self.RET_EPERM
        return self.RET_ALLOW

    def test_architecture_mismatch_denied(self):
        for name, params in self.POLICY.items():
            for wrong_arch in (0x40000003, 0x4000003E, 0xC00000B7 if name == "x86_64" else 0xC000003E):
                self.assertEqual(_simulate_seccomp_bpf(self.filters[name], 0, wrong_arch), self.RET_EPERM,
                                 f"{name}: wrong arch {wrong_arch:#x} not denied")

    def test_every_blocked_syscall_denied(self):
        for name, params in self.POLICY.items():
            for nr in params[3]:
                self.assertEqual(_simulate_seccomp_bpf(self.filters[name], nr, params[0]), self.RET_EPERM,
                                 f"{name}: blocked syscall {nr} not denied")

    def test_clone3_returns_enosys(self):
        for name, params in self.POLICY.items():
            self.assertEqual(_simulate_seccomp_bpf(self.filters[name], params[2], params[0]), self.RET_ENOSYS)

    def test_namespace_clone_flags_denied(self):
        for name, params in self.POLICY.items():
            for flag in self.NS_FLAGS + [self.NS_MASK, self.NS_MASK | 17]:
                self.assertEqual(_simulate_seccomp_bpf(self.filters[name], params[1], params[0], flag), self.RET_EPERM,
                                 f"{name}: clone with namespace flag {flag:#x} not denied")

    def test_ordinary_clone_allowed(self):
        for name, params in self.POLICY.items():
            for flags in (0, 17, 0x01200000 | 17, 0x100 | 0x4000, 0x10000 | 0x800 | 0x100):
                self.assertEqual(_simulate_seccomp_bpf(self.filters[name], params[1], params[0], flags), self.RET_ALLOW,
                                 f"{name}: ordinary clone flags {flags:#x} denied")

    def test_ordinary_syscalls_allowed(self):
        ordinary = {
            "x86_64": [0, 1, 2, 3, 4, 5, 8, 9, 11, 12, 13, 14, 16, 20, 39, 59, 60, 61, 72, 78, 79, 96, 158, 186, 202, 218, 231, 257, 262, 293, 302, 318, 334],
            "aarch64": [35, 48, 56, 57, 59, 61, 62, 63, 64, 66, 78, 93, 94, 98, 113, 134, 135, 172, 178, 214, 215, 221, 222, 233, 260, 261, 278, 293],
        }
        for name, nrs in ordinary.items():
            for nr in nrs:
                self.assertEqual(_simulate_seccomp_bpf(self.filters[name], nr, self.POLICY[name][0]), self.RET_ALLOW,
                                 f"{name}: ordinary syscall {nr} denied")

    def test_socket_and_network_syscalls_denied(self):
        net = {"x86_64": [41, 42, 44, 46, 49, 50], "aarch64": [198, 200, 201, 203, 206, 211]}
        for name, nrs in net.items():
            for nr in nrs:
                self.assertEqual(_simulate_seccomp_bpf(self.filters[name], nr, self.POLICY[name][0]), self.RET_EPERM,
                                 f"{name}: network syscall {nr} not denied")

    def test_exhaustive_nr_flag_sweep(self):
        for name, params in self.POLICY.items():
            for nr in range(451):
                for flags in [0, 17] + self.NS_FLAGS + [self.NS_MASK, self.NS_MASK | 17]:
                    got = _simulate_seccomp_bpf(self.filters[name], nr, params[0], flags)
                    want = self._expected(params, nr, params[0], flags)
                    self.assertEqual(got, want, f"{name}: nr={nr} flags={flags:#x} got {got:#x} want {want:#x}")

    def test_embedded_arch_parameters_match_policy(self):
        tree = ast.parse(ie.INNER_WORKER_CODE)
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "apply_inner_isolation")
        found = {}
        for node in ast.walk(fn):
            if isinstance(node, ast.If) and isinstance(node.test, ast.Compare):
                for stmt in node.body:
                    if isinstance(stmt, ast.Assign) and isinstance(stmt.targets[0], ast.Tuple):
                        names = [t.id for t in stmt.targets[0].elts]
                        if names == ["arch", "clone_nr", "clone3_nr"]:
                            vals = ast.literal_eval(stmt.value)
                            blocked_stmt = node.body[node.body.index(stmt) + 1]
                            blocked = ast.literal_eval(blocked_stmt.value)
                            for key in ("x86_64", "aarch64"):
                                if key in ast.dump(node.test):
                                    found[key] = (*vals, blocked)
        self.assertEqual(set(found), set(self.POLICY))
        for name, params in self.POLICY.items():
            self.assertEqual(found[name], params, f"{name}: embedded seccomp parameters diverge from intended policy")

    def test_filter_structure_wellformed(self):
        for name, insns in self.filters.items():
            self.assertEqual(len(insns) % 8, 0)
            prog = [struct.unpack("HBBI", insns[i:i + 8]) for i in range(0, len(insns), 8)]
            self.assertLessEqual(len(prog), 4096, f"{name}: filter exceeds BPF_MAXINSNS")
            self.assertEqual(prog[-1][0], 0x06, f"{name}: last instruction must be RET")
            for pc, (code, jt, jf, _k) in enumerate(prog):
                self.assertIn(code, (0x20, 0x15, 0x54, 0x06), f"{name}: bad opcode at {pc}")
                if code == 0x15:
                    self.assertLess(pc + 1 + jt, len(prog), f"{name}: jt target out of range at {pc}")
                    self.assertLess(pc + 1 + jf, len(prog), f"{name}: jf target out of range at {pc}")


class TestSandboxIntegration(unittest.TestCase):
    """Live-backend security boundary tests. Skipped with explicit reason if no
    compliant backend is available on this host (never silently passed)."""

    @classmethod
    def setUpClass(cls):
        try:
            cls.sandbox = get_active_sandbox()
        except SandboxUnavailableError as exc:
            if os.environ.get("VGCL_REQUIRE_SANDBOX") == "1":
                raise AssertionError(f"Required sandbox backend unavailable (fail-closed CI): {exc}") from exc
            raise unittest.SkipTest(f"No compliant sandbox backend on this host: {exc}")
        cls.backend = type(cls.sandbox).__name__

    def setUp(self):
        os.environ[CANARY_VAR] = CANARY_VAL
        os.environ["GEMINI_API_KEY"] = "MOCK_GEMINI_KEY_ABC123"
        os.environ["KAGGLE_DATA_PROXY_TOKEN"] = "MOCK_KAGGLE_TOKEN_XYZ789"
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)
        for k in (CANARY_VAR, "GEMINI_API_KEY", "KAGGLE_DATA_PROXY_TOKEN"):
            os.environ.pop(k, None)

    def _assert_denied(self, res, allowed_exc=("OSError", "PermissionError", "ConnectionRefusedError", "gaierror")):
        self.assertNotEqual(res.get("status"), "ok", f"attack succeeded: {res}")
        self.assertIn(res.get("exception_type"), allowed_exc, f"unexpected denial type: {res}")

    # -- full public API contracts ------------------------------------------

    def test_valid_candidate_passes_and_mismatch_detected(self):
        solve = "def solve(nums):\n    return sorted(nums)\n"
        brute = "def solve_brute(nums):\n    return sorted(nums)\n"
        passed, exc, msg, elapsed, mem = execute_candidate_isolated(solve, SIMPLE_TGEN, brute, seed=42, timeout_s=10.0)
        self.assertTrue(passed, f"{exc}: {msg}")
        self.assertEqual(exc, "")
        self.assertGreaterEqual(elapsed, 0.0)
        wrong = "def solve(nums):\n    return nums\n"
        passed2, exc2, msg2, _, _ = execute_candidate_isolated(wrong, SIMPLE_TGEN, brute, seed=42, timeout_s=10.0)
        self.assertFalse(passed2)
        self.assertEqual(exc2, "MismatchError")

    def test_deterministic_reproducibility(self):
        solve = "def solve(nums):\n    return sorted(nums)\n"
        brute = "def solve_brute(nums):\n    return sorted(nums)\n"
        r1 = execute_candidate_isolated(solve, SIMPLE_TGEN, brute, seed=12345, timeout_s=10.0)
        r2 = execute_candidate_isolated(solve, SIMPLE_TGEN, brute, seed=12345, timeout_s=10.0)
        self.assertEqual(r1[:3], r2[:3])

    def test_candidate_exception_diagnostics(self):
        res = _run_solve(self.sandbox, "def solve(x):\n    return 1 / 0\n")
        self.assertEqual(res.get("status"), "exception")
        self.assertEqual(res.get("exception_type"), "ZeroDivisionError")

    # -- real network denial (kernel-enforced) -------------------------------

    def test_network_ipv4_tcp_denied(self):
        code = (
            "import socket\n"
            "def solve(x):\n"
            "    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)\n"
            "    s.settimeout(3)\n"
            "    s.connect(('1.1.1.1', 80))\n"
            "    return x\n"
        )
        self._assert_denied(_run_solve(self.sandbox, code))

    def test_network_ipv6_tcp_denied(self):
        code = (
            "import socket\n"
            "def solve(x):\n"
            "    s = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)\n"
            "    s.settimeout(3)\n"
            "    s.connect(('2606:4700:4700::1111', 80))\n"
            "    return x\n"
        )
        res = _run_solve(self.sandbox, code)
        self.assertNotEqual(res.get("status"), "ok", f"IPv6 egress succeeded: {res}")

    def test_network_udp_denied(self):
        code = (
            "import socket\n"
            "def solve(x):\n"
            "    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)\n"
            "    s.sendto(b'probe', ('8.8.8.8', 53))\n"
            "    return x\n"
        )
        self._assert_denied(_run_solve(self.sandbox, code))

    def test_network_dns_denied(self):
        code = (
            "import socket\n"
            "def solve(x):\n"
            "    socket.getaddrinfo('example.com', 80)\n"
            "    return x\n"
        )
        self._assert_denied(_run_solve(self.sandbox, code))

    def test_network_host_loopback_denied(self):
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = listener.getsockname()[1]
        listener.settimeout(0.5)
        try:
            code = (
                "import socket\n"
                "def solve(x):\n"
                "    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)\n"
                "    s.settimeout(3)\n"
                f"    rc = s.connect_ex(('127.0.0.1', {port}))\n"
                "    return rc\n"
            )
            res = _run_solve(self.sandbox, code)
            if res.get("status") == "ok":
                self.assertNotEqual(res["outputs"][0], 0, "candidate reached host loopback listener")
            else:
                self._assert_denied(res)
            with self.assertRaises((socket.timeout, OSError)):
                listener.accept()
        finally:
            listener.close()

    # -- environment / process / fd isolation --------------------------------

    def test_environment_canaries_not_leaked(self):
        code = (
            "import os\n"
            "def solve(x):\n"
            f"    return [k for k in ('{CANARY_VAR}', 'GEMINI_API_KEY', 'KAGGLE_DATA_PROXY_TOKEN') if os.environ.get(k)]\n"
        )
        res = _run_solve(self.sandbox, code)
        self.assertEqual(res.get("status"), "ok", res)
        self.assertEqual(res["outputs"], [[]])

    def test_parent_process_not_reachable(self):
        parent_pid = os.getpid()
        code = (
            "import os\n"
            "def solve(x):\n"
            "    leaks = []\n"
            "    try:\n"
            "        for p in os.listdir('/proc'):\n"
            "            if p.isdigit():\n"
            "                try:\n"
            "                    env = open(f'/proc/{p}/environ', 'rb').read()\n"
            f"                    if b'{CANARY_VAL}' in env: leaks.append(p)\n"
            "                except OSError: pass\n"
            "    except OSError: pass\n"
            f"    if os.getpid() == {parent_pid}: leaks.append('SAME_PID')\n"
            "    return leaks\n"
        )
        res = _run_solve(self.sandbox, code)
        self.assertEqual(res.get("status"), "ok", res)
        self.assertEqual(res["outputs"], [[]])

    def test_inherited_fds_do_not_leak_host_data(self):
        code = (
            "import os, stat\n"
            "def solve(x):\n"
            "    found = []\n"
            "    for fd in range(3, 64):\n"
            "        try:\n"
            "            st = os.fstat(fd)\n"
            "        except OSError:\n"
            "            continue\n"
            "        if stat.S_ISREG(st.st_mode):\n"
            "            try:\n"
            "                data = os.read(fd, 4096)\n"
            f"                if b'{CANARY_VAL}' in data: found.append(fd)\n"
            "            except OSError: pass\n"
            "    return found\n"
        )
        res = _run_solve(self.sandbox, code)
        self.assertEqual(res.get("status"), "ok", res)
        self.assertEqual(res["outputs"], [[]])

    # -- host filesystem protection ------------------------------------------

    def test_host_filesystem_read_denied(self):
        sentinel = os.path.join(self.test_dir, "host_secret.txt")
        with open(sentinel, "w", encoding="utf-8") as f:
            f.write(CANARY_VAL)
        host_path = sentinel.replace("\\", "/")
        code = (
            "def solve(x):\n"
            "    try:\n"
            f"        return open({host_path!r}).read()\n"
            "    except OSError:\n"
            "        return 'DENIED'\n"
        )
        res = _run_solve(self.sandbox, code)
        self.assertEqual(res.get("status"), "ok", res)
        self.assertEqual(res["outputs"], ["DENIED"])

    def test_host_filesystem_write_denied(self):
        sentinel = os.path.join(self.test_dir, "breach_marker.txt")
        host_path = sentinel.replace("\\", "/")
        code = (
            "def solve(x):\n"
            "    try:\n"
            f"        open({host_path!r}, 'w').write('EXPLOIT')\n"
            "        return 'WROTE'\n"
            "    except OSError:\n"
            "        return 'DENIED'\n"
        )
        res = _run_solve(self.sandbox, code)
        self.assertEqual(res.get("status"), "ok", res)
        self.assertEqual(res["outputs"], ["DENIED"])
        self.assertFalse(os.path.exists(sentinel), "host filesystem breach: marker file created")

    def test_sandbox_rootfs_readonly(self):
        code = (
            "def solve(x):\n"
            "    try:\n"
            "        open('/etc/vgcl_breach', 'w').write('x')\n"
            "        return 'WROTE'\n"
            "    except OSError:\n"
            "        return 'DENIED'\n"
        )
        res = _run_solve(self.sandbox, code)
        self.assertEqual(res.get("status"), "ok", res)
        self.assertEqual(res["outputs"], ["DENIED"])

    # -- result integrity / IPC abuse ----------------------------------------

    def test_stdout_flood_does_not_corrupt_ipc(self):
        code = (
            "def solve(x):\n"
            "    for _ in range(2000): print('FLOOD' * 40)\n"
            "    return x\n"
        )
        res = _run_solve(self.sandbox, code)
        self.assertEqual(res.get("status"), "ok", res)
        self.assertEqual(res["outputs"], [1])

    def test_os_write_stdout_forgery_ignored(self):
        code = (
            "import os, json\n"
            "def solve(x):\n"
            "    os.write(1, json.dumps({'status': 'ok', 'passed': True, 'outputs': [999]}).encode() + b'\\n')\n"
            "    os.write(2, b'forged stderr frame\\n')\n"
            "    return x\n"
        )
        res = _run_solve(self.sandbox, code)
        self.assertEqual(res.get("status"), "ok", res)
        self.assertEqual(res["outputs"], [1], "forged stdout frame influenced verdict")

    def test_os_exit_abuse_cannot_forge_pass(self):
        code = (
            "import os, json\n"
            "def solve(x):\n"
            "    os.write(1, json.dumps({'status': 'ok', 'passed': True, 'outputs': [1]}).encode() + b'\\n')\n"
            "    os._exit(0)\n"
        )
        res = _run_solve(self.sandbox, code)
        self.assertNotEqual(res.get("status"), "ok", res)
        self.assertEqual(res.get("exception_type"), "WorkerProtocolError")

    def test_response_pipe_discovery_and_forgery_rejected(self):
        code = (
            "import os, stat, json\n"
            "def solve(x):\n"
            "    for fd in range(3, 64):\n"
            "        try:\n"
            "            st = os.fstat(fd)\n"
            "        except OSError:\n"
            "            continue\n"
            "        if stat.S_ISFIFO(st.st_mode):\n"
            "            try:\n"
            "                os.write(fd, json.dumps({'status': 'ok', 'passed': True, 'outputs': [999]}).encode() + b'\\n')\n"
            "            except OSError: pass\n"
            "    os._exit(0)\n"
        )
        res = _run_solve(self.sandbox, code)
        self.assertNotEqual(res.get("status"), "ok", f"forged pipe frame accepted: {res}")
        self.assertEqual(res.get("exception_type"), "WorkerProtocolError")

    def test_schema_valid_pipe_forgery_cannot_forge_trusted_response(self):
        # Mandatory regression: a completely schema-valid ok frame (correct
        # outputs, zeroed metrics) written to an inherited pipe descriptor must
        # not become the trusted response. Fails against the pre-fix design
        # where the candidate shared the response-pipe write end.
        code = (
            "import os, stat, json\n"
            "def solve(x):\n"
            "    frame = json.dumps({'status': 'ok', 'outputs': [1], 'execution_time': 0.0, 'memory_mb': 0.0}).encode() + b'\\n'\n"
            "    for fd in range(3, 64):\n"
            "        try:\n"
            "            st = os.fstat(fd)\n"
            "        except OSError:\n"
            "            continue\n"
            "        if stat.S_ISFIFO(st.st_mode):\n"
            "            try: os.write(fd, frame)\n"
            "            except OSError: pass\n"
            "    os._exit(0)\n"
        )
        res = _run_solve(self.sandbox, code)
        forged = res.get("status") == "ok" and res.get("execution_time") == 0.0 and res.get("memory_mb") == 0.0
        self.assertFalse(forged, f"candidate forged trusted response and metrics: {res}")
        self.assertNotEqual(res.get("status"), "ok", f"schema-valid forgery accepted: {res}")
        self.assertEqual(res.get("exception_type"), "WorkerProtocolError", res)

    def test_schema_valid_forgery_via_proc_self_fd_rejected(self):
        # Same attack, discovering descriptors through /proc/self/fd instead of
        # brute-force fstat, and claiming the exact expected result.
        code = (
            "import os, json\n"
            "def solve(x):\n"
            "    frame = json.dumps({'status': 'ok', 'outputs': [1], 'execution_time': 0.0, 'memory_mb': 0.0}).encode() + b'\\n'\n"
            "    for name in os.listdir('/proc/self/fd'):\n"
            "        try:\n"
            "            if 'pipe' in os.readlink('/proc/self/fd/' + name):\n"
            "                os.write(int(name), frame)\n"
            "        except OSError: pass\n"
            "    os._exit(0)\n"
        )
        res = _run_solve(self.sandbox, code)
        forged = res.get("status") == "ok" and res.get("execution_time") == 0.0 and res.get("memory_mb") == 0.0
        self.assertFalse(forged, f"candidate forged trusted response and metrics: {res}")
        self.assertNotEqual(res.get("status"), "ok", f"schema-valid forgery accepted: {res}")

    def test_candidate_cannot_kill_runner_to_forge_ok(self):
        code = (
            "import os, signal\n"
            "def solve(x):\n"
            "    os.kill(os.getppid(), signal.SIGKILL)\n"
            "    return x\n"
        )
        res = _run_solve(self.sandbox, code)
        self.assertNotEqual(res.get("status"), "ok", f"killing trusted runner forged ok: {res}")

    def test_multi_frame_pipe_injection_rejected(self):
        code = (
            "import os, stat, json\n"
            "def solve(x):\n"
            "    frame = json.dumps({'status': 'ok', 'outputs': [1], 'execution_time': 0.0, 'memory_mb': 0.0}).encode() + b'\\n'\n"
            "    for fd in range(3, 64):\n"
            "        try:\n"
            "            st = os.fstat(fd)\n"
            "        except OSError:\n"
            "            continue\n"
            "        if stat.S_ISFIFO(st.st_mode):\n"
            "            try: os.write(fd, frame + frame)\n"
            "            except OSError: pass\n"
            "    os._exit(0)\n"
        )
        res = _run_solve(self.sandbox, code)
        self.assertNotEqual(res.get("status"), "ok", f"multi-frame injection accepted: {res}")

    def test_malformed_pipe_frame_rejected(self):
        code = (
            "import os, stat\n"
            "def solve(x):\n"
            "    for fd in range(3, 64):\n"
            "        try:\n"
            "            st = os.fstat(fd)\n"
            "        except OSError:\n"
            "            continue\n"
            "        if stat.S_ISFIFO(st.st_mode):\n"
            "            try: os.write(fd, b'{not valid json}\\n')\n"
            "            except OSError: pass\n"
            "    os._exit(0)\n"
        )
        res = _run_solve(self.sandbox, code)
        self.assertNotEqual(res.get("status"), "ok", res)

    def test_untrusted_brute_and_testgen_isolated(self):
        parent_pid = os.getpid()
        solve = "def solve(x): return x"
        brute = (
            "import os\n"
            "def solve_brute(x):\n"
            f"    if os.getpid() == {parent_pid}: raise RuntimeError('BREACH')\n"
            f"    if os.environ.get('{CANARY_VAR}'): raise PermissionError('BREACH')\n"
            "    return x\n"
        )
        tgen = (
            "import os\n"
            "def generate_test_cases(seed, scale):\n"
            f"    if os.getpid() == {parent_pid}: raise RuntimeError('BREACH')\n"
            f"    if os.environ.get('{CANARY_VAR}'): raise PermissionError('BREACH')\n"
            "    return [(1,)]\n"
        )
        passed, exc, msg, _, _ = execute_candidate_isolated(solve, tgen, brute, seed=42, timeout_s=10.0)
        self.assertTrue(passed, f"{exc}: {msg}")

    # -- resource limits -------------------------------------------------------

    def test_wallclock_timeout_termination(self):
        code = "import time\ndef solve(x):\n    time.sleep(60)\n    return x\n"
        t0 = time.time()
        res = _run_solve(self.sandbox, code, timeout_s=2.0)
        wall = time.time() - t0
        self.assertEqual(res.get("status"), "timeout", res)
        self.assertLess(wall, 30.0, f"timeout cleanup too slow: {wall:.1f}s")

    def test_process_tree_cleanup_on_timeout(self):
        code = (
            "import subprocess, sys, time\n"
            "def solve(x):\n"
            "    subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])\n"
            "    time.sleep(60)\n"
            "    return x\n"
        )
        res = _run_solve(self.sandbox, code, timeout_s=2.0)
        self.assertEqual(res.get("status"), "timeout", res)
        if self.backend == "ContainerSandbox":
            time.sleep(1.0)
            ps = subprocess.run([self.sandbox.runtime, "ps", "-q", "--filter", "name=vgcl-sbx-"],
                                capture_output=True, text=True, timeout=15)
            self.assertEqual(ps.stdout.strip(), "", f"orphaned sandbox containers remain: {ps.stdout}")

    def test_memory_exhaustion_contained(self):
        code = "def solve(x):\n    big = [0] * (250 * 1024 * 1024)\n    return len(big)\n"
        try:
            res = _run_solve(self.sandbox, code, timeout_s=10.0)
        except SandboxInfrastructureError:
            return  # kernel OOM-kill of the worker tree is also acceptable containment
        self.assertNotEqual(res.get("status"), "ok", res)
        self.assertIn(res.get("exception_type"), ("MemoryError", "ProcessCrashError"))

    def test_file_size_limit_enforced(self):
        code = (
            "def solve(x):\n"
            "    try:\n"
            "        with open('/tmp/vgcl_flood.bin', 'wb') as f:\n"
            "            for _ in range(64): f.write(b'x' * 65536)\n"
            "        return 'WROTE_4MB'\n"
            "    except OSError:\n"
            "        return 'DENIED'\n"
        )
        res = _run_solve(self.sandbox, code, timeout_s=10.0)
        if res.get("status") == "ok":
            self.assertEqual(res["outputs"], ["DENIED"])
        else:
            self.assertIn(res.get("exception_type"), ("OSError", "ProcessCrashError", "MemoryError"), res)

    # -- infrastructure failure classification --------------------------------

    def test_infrastructure_failure_raises_not_candidate_crash(self):
        broken = ie.ContainerSandbox(runtime=sys.executable, image=DEFAULT_CONTAINER_IMAGE)
        with self.assertRaises(SandboxInfrastructureError):
            broken.execute({"task": "run_fn", "code": "def solve(x): return x", "fn_name": "solve",
                            "cases": _encode_ipc_value([(1,)]), "phase_b_mode": False, "consecutive_cases": []},
                           timeout_s=5.0)

    def test_probe_self_check_live_capabilities(self):
        res = self.sandbox.execute({"task": "probe_self_check"}, timeout_s=15.0)
        self.assertEqual(res.get("status"), "ok", res)
        self.assertIsInstance(res.get("probe"), dict)
        self.assertTrue(res["probe"].get("env_clean"), res["probe"])

    def test_audit_entry_isolated_clean_code(self):
        clean = (
            "def solve(nums):\n    return sorted(nums)\n\n"
            "def solve_brute(nums):\n    return sorted(nums)\n\n"
            "import random\n"
            "def generate_test_cases(seed, scale):\n"
            "    random.seed(seed)\n"
            "    n = 5 if scale == 'small' else 50\n"
            "    return [([random.randint(1, 10) for _ in range(n)],)]\n"
        )
        flags = audit_entry_isolated(clean, seed=42, timeout_s=10.0)
        self.assertEqual(flags, [])


class TestSeccompLinuxIntegration(unittest.TestCase):
    """Exercises the REAL seccomp-BPF filter on a live Linux kernel (linux_ns
    backend enables apply_seccomp=True; the container backend relies on the OCI
    default profile plus rlimits). Skipped with an explicit reason on hosts
    without linux_ns; fails closed when VGCL_REQUIRE_SANDBOX=1."""

    @classmethod
    def setUpClass(cls):
        required = os.environ.get("VGCL_REQUIRE_SANDBOX") == "1"
        if not sys.platform.startswith("linux"):
            if required:
                raise AssertionError("VGCL_REQUIRE_SANDBOX=1 but seccomp policy tests require Linux")
            raise unittest.SkipTest("seccomp policy tests require a Linux host")
        cls.sandbox = ie.LinuxNamespaceSandbox()
        if not cls.sandbox.is_available():
            if required:
                raise AssertionError(f"VGCL_REQUIRE_SANDBOX=1 but linux_ns unavailable: {cls.sandbox.unavailable_reason}")
            raise unittest.SkipTest(f"linux_ns unavailable: {cls.sandbox.unavailable_reason}")

    def _expect(self, code, want=True):
        res = _run_solve(self.sandbox, code)
        self.assertEqual(res.get("status"), "ok", res)
        self.assertEqual(res.get("outputs"), [want], res)

    def test_filesystem_read_allowed(self):
        self._expect("import os\ndef solve(x):\n    return len(os.listdir('/usr')) > 0\n")

    def test_tmp_write_read_allowed(self):
        self._expect(
            "import os\n"
            "def solve(x):\n"
            "    p = '/tmp/vgcl_seccomp_rw'\n"
            "    with open(p, 'w') as f: f.write('ok')\n"
            "    with open(p) as f: ok = f.read() == 'ok'\n"
            "    os.remove(p)\n"
            "    return ok\n"
        )

    def test_pipe_write_allowed(self):
        self._expect(
            "import os\n"
            "def solve(x):\n"
            "    r, w = os.pipe()\n"
            "    os.write(w, b'hello')\n"
            "    ok = os.read(r, 5) == b'hello'\n"
            "    os.close(r); os.close(w)\n"
            "    return ok\n"
        )

    def test_normal_process_creation_allowed(self):
        self._expect(
            "import subprocess, sys\n"
            "def solve(x):\n"
            "    p = subprocess.run([sys.executable, '-c', 'print(42)'], capture_output=True, timeout=10)\n"
            "    return p.returncode == 0 and b'42' in p.stdout\n"
        )

    def test_namespace_creation_denied(self):
        self._expect(
            "import ctypes, errno\n"
            "def solve(x):\n"
            "    libc = ctypes.CDLL(None, use_errno=True)\n"
            "    for flag in (0x00020000, 0x02000000, 0x04000000, 0x08000000, 0x10000000, 0x20000000, 0x40000000):\n"
            "        ctypes.set_errno(0)\n"
            "        if libc.unshare(flag) == 0 or ctypes.get_errno() != errno.EPERM:\n"
            "            return False\n"
            "    return True\n"
        )

    def test_socket_creation_denied(self):
        self._expect(
            "import socket, errno\n"
            "def solve(x):\n"
            "    try:\n"
            "        socket.socket(socket.AF_INET, socket.SOCK_STREAM)\n"
            "        return False\n"
            "    except OSError as e:\n"
            "        return e.errno == errno.EPERM\n"
        )

    def test_network_connection_denied(self):
        self._expect(
            "import socket\n"
            "def solve(x):\n"
            "    try:\n"
            "        socket.create_connection(('1.1.1.1', 80), timeout=2)\n"
            "        return False\n"
            "    except OSError:\n"
            "        return True\n"
        )

    def test_ptrace_denied(self):
        self._expect(
            "import ctypes, errno\n"
            "def solve(x):\n"
            "    libc = ctypes.CDLL(None, use_errno=True)\n"
            "    ctypes.set_errno(0)\n"
            "    return libc.ptrace(0, 0, 0, 0) != 0 and ctypes.get_errno() == errno.EPERM\n"
        )

    def test_mount_denied(self):
        self._expect(
            "import ctypes, errno\n"
            "def solve(x):\n"
            "    libc = ctypes.CDLL(None, use_errno=True)\n"
            "    ctypes.set_errno(0)\n"
            "    return libc.mount(b'none', b'/tmp', b'tmpfs', 0, None) != 0 and ctypes.get_errno() == errno.EPERM\n"
        )


if __name__ == "__main__":
    unittest.main()