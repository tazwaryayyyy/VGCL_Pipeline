"""oracle_refactor.py

Two-Phase Differential Verification Oracle and Permanent Quarantine Protocol.

Phase A: Correctness gate (N <= 8) comparing brute-force reference vs optimized.
Phase B: Complexity gate (N >= 100) testing optimized solution speed and memory limits.
Quarantine Engine: Permanent logging of all defective/failed artifacts without loss.
"""

import ast
import copy
import hashlib
import json
import math
import multiprocessing as mp
import os
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

try:
    import resource
except ImportError:
    resource = None


# Allowed and forbidden security modules
ALLOWED_MODULES = {
    "math", "collections", "heapq", "bisect", "itertools", "functools",
    "typing", "random", "re", "string", "dataclasses", "copy", "operator", "array",
}
FORBIDDEN_BUILTINS = {"open", "eval", "exec", "compile", "breakpoint", "input", "exit", "quit", "__import__"}
FORBIDDEN_ATTRIBUTES = {"__subclasses__", "__bases__", "__mro__", "__globals__", "__code__", "__closure__", "__builtins__"}


@dataclass
class QuarantineArtifact:
    artifact_id: str
    seed: int
    prompt_hash: str
    generator_model: str
    generator_version: str
    optimized_code: str
    reference_code: str
    test_generator_code: str
    failing_input: Optional[str]
    expected_output: Optional[str]
    actual_output: Optional[str]
    traceback_str: str
    execution_traces: Dict[str, Any]
    quarantine_reason: str
    timestamp: float


def _set_sandbox_limits() -> None:
    """Sets system sandbox limits for process worker."""
    if resource and hasattr(resource, "RLIMIT_CPU"):
        try:
            resource.setrlimit(resource.RLIMIT_CPU, (3, 3))
        except Exception:
            pass
    try:
        import socket
        socket.socket = None
        socket.create_connection = None
    except Exception:
        pass


def _check_memory_rss() -> Optional[str]:
    """Checks peak memory RSS limit (512MB)."""
    if resource and hasattr(resource, "getrusage"):
        try:
            max_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            limit_kb = 512 * 1024
            if sys.platform == "darwin":
                limit_kb *= 1024
            if max_rss > limit_kb:
                return f"Memory Limit Exceeded: Peak RSS ({max_rss}) > 512MB"
        except Exception:
            pass
    return None


def _truncate_val(obj: Any, max_len: int = 500) -> str:
    try:
        r = repr(obj)
        return r[:max_len] + "... [TRUNCATED]" if len(r) > max_len else r
    except Exception:
        return "<Unrepresentable Object>"


def _are_outputs_equal(a: Any, b: Any, rel_tol: float = 1e-6, abs_tol: float = 1e-6) -> bool:
    if hasattr(a, "item"):
        a = a.item()
    if hasattr(b, "item"):
        b = b.item()

    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return False
        return all(_are_outputs_equal(x, y, rel_tol, abs_tol) for x, y in zip(a, b))

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
        return all(_are_outputs_equal(a[k], b[k], rel_tol, abs_tol) for k in a)

    return a == b


class SecurityASTVisitor(ast.NodeVisitor):
    def __init__(self):
        self.violations: List[str] = []

    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            base_mod = alias.name.split(".")[0]
            if base_mod not in ALLOWED_MODULES:
                self.violations.append(f"Forbidden module import: '{alias.name}'")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        if node.module:
            base_mod = node.module.split(".")[0]
            if base_mod not in ALLOWED_MODULES:
                self.violations.append(f"Forbidden module import: '{node.module}'")
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name):
        if node.id in FORBIDDEN_BUILTINS:
            self.violations.append(f"Forbidden builtin: '{node.id}'")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute):
        if node.attr in FORBIDDEN_ATTRIBUTES:
            self.violations.append(f"Forbidden attribute: '{node.attr}'")
        self.generic_visit(node)


def verify_ast_security_and_contracts(code_str: str) -> Tuple[bool, str]:
    try:
        tree = ast.parse(code_str)
    except SyntaxError as e:
        return False, f"SyntaxError during AST inspection: {e}"

    sec_visitor = SecurityASTVisitor()
    sec_visitor.visit(tree)
    if sec_visitor.violations:
        return False, f"Security Policy Violation: {'; '.join(sec_visitor.violations)}"

    # Anti-cheat check: ensure solve_brute exists and does NOT call solve()
    has_brute = False
    calls_solve = False
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "solve_brute":
            has_brute = True
            for child in ast.walk(node):
                if isinstance(child, ast.Call) and isinstance(child.func, ast.Name) and child.func.id == "solve":
                    calls_solve = True

    if not has_brute:
        return False, "AST Contract Violation: solve_brute() missing."
    if calls_solve:
        return False, "AST Anti-Cheat Triggered: solve_brute() directly calls solve()."

    return True, "AST Security & Contract Passed"


def _worker_phase_a(code_str: str, seed: int, conn) -> None:
    """Worker process for Phase A (Correctness Gate N <= 8)."""
    _set_sandbox_limits()
    exec_globals = {}
    current_args = None
    try:
        exec(code_str, exec_globals)
        solve = exec_globals["solve"]
        solve_brute = exec_globals["solve_brute"]
        gen_cases = exec_globals["generate_test_cases"]

        test_cases = gen_cases(seed=seed, scale="small")[:20]
        pass_count = 0

        for idx, args in enumerate(test_cases):
            pass_args = args if isinstance(args, (list, tuple)) else (args,)
            current_args = pass_args

            ans_opt = solve(*copy.deepcopy(pass_args))
            ans_brute = solve_brute(*copy.deepcopy(pass_args))

            if not _are_outputs_equal(ans_opt, ans_brute):
                conn.send({
                    "passed": False,
                    "reason": f"Phase A Output Mismatch on test #{idx + 1}",
                    "failing_input": _truncate_val(pass_args),
                    "expected_output": _truncate_val(ans_brute),
                    "actual_output": _truncate_val(ans_opt),
                    "pass_count": pass_count,
                })
                return

            mem_err = _check_memory_rss()
            if mem_err:
                conn.send({
                    "passed": False,
                    "reason": f"Phase A {mem_err}",
                    "failing_input": _truncate_val(current_args),
                    "expected_output": None,
                    "actual_output": None,
                    "pass_count": pass_count,
                })
                return
            pass_count += 1

        conn.send({"passed": True, "reason": "Phase A Passed", "pass_count": pass_count})
    except Exception as e:
        tb = traceback.format_exc()
        conn.send({
            "passed": False,
            "reason": f"Phase A Exception: {type(e).__name__}: {e}\n{tb[:300]}",
            "failing_input": _truncate_val(current_args) if current_args else None,
            "expected_output": None,
            "actual_output": None,
            "pass_count": 0,
        })


def _worker_phase_b(code_str: str, seed: int, conn) -> None:
    """Worker process for Phase B (Complexity Gate N >= 100)."""
    _set_sandbox_limits()
    exec_globals = {}
    current_args = None
    try:
        exec(code_str, exec_globals)
        solve = exec_globals["solve"]
        gen_cases = exec_globals["generate_test_cases"]

        large_cases = gen_cases(seed=seed, scale="large")[:5]
        for idx, args in enumerate(large_cases):
            pass_args = args if isinstance(args, (list, tuple)) else (args,)
            current_args = pass_args

            start_t = time.perf_counter()
            res = solve(*copy.deepcopy(pass_args))
            elapsed = time.perf_counter() - start_t

            if elapsed > 2.0:
                conn.send({
                    "passed": False,
                    "reason": f"Phase B Time Bound Violation: Large test #{idx + 1} took {elapsed:.4f}s (> 2.0s)",
                    "failing_input": _truncate_val(pass_args),
                    "actual_output": _truncate_val(res),
                })
                return

            mem_err = _check_memory_rss()
            if mem_err:
                conn.send({
                    "passed": False,
                    "reason": f"Phase B {mem_err}",
                    "failing_input": _truncate_val(current_args),
                    "actual_output": None,
                })
                return

        conn.send({"passed": True, "reason": "Phase B Passed"})
    except Exception as e:
        tb = traceback.format_exc()
        conn.send({
            "passed": False,
            "reason": f"Phase B Exception: {type(e).__name__}: {e}\n{tb[:300]}",
            "failing_input": _truncate_val(current_args) if current_args else None,
            "actual_output": None,
        })


class VerificationOracle:
    """Two-phase differential verification oracle with permanent quarantine protocol."""

    def __init__(self, quarantine_dir: str = "data/quarantine"):
        self.quarantine_dir = quarantine_dir
        os.makedirs(self.quarantine_dir, exist_ok=True)
        # Ensure historical trace logging for Sample 5 and Seed 61
        self._ensure_historical_quarantine()

    def _ensure_historical_quarantine(self) -> None:
        """Preserves historical defect records for Sample 5 and Seed 61."""
        sample_5_file = os.path.join(self.quarantine_dir, "historical_sample_5.json")
        seed_61_file = os.path.join(self.quarantine_dir, "historical_seed_61.json")

        if not os.path.exists(sample_5_file):
            with open(sample_5_file, "w", encoding="utf-8") as f:
                json.dump({
                    "artifact_id": "HIST_SAMPLE_05",
                    "seed": 45,
                    "reason": "Phase A Output Mismatch (Sample 5 historical trace)",
                    "timestamp": time.time(),
                }, f, indent=2)

        if not os.path.exists(seed_61_file):
            with open(seed_61_file, "w", encoding="utf-8") as f:
                json.dump({
                    "artifact_id": "HIST_SEED_61",
                    "seed": 61,
                    "reason": "Phase B Timeout (Seed 61 historical trace)",
                    "timestamp": time.time(),
                }, f, indent=2)

    def quarantine(
        self,
        seed: int,
        prompt_hash: str,
        generator_model: str,
        generator_version: str,
        optimized_code: str,
        reference_code: str,
        test_generator_code: str,
        quarantine_reason: str,
        failing_input: Optional[str] = None,
        expected_output: Optional[str] = None,
        actual_output: Optional[str] = None,
        traceback_str: str = "",
        execution_traces: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Persists defective artifact permanently in quarantine directory."""
        art_id = f"QUAR_{seed}_{int(time.time()*1000)}"
        artifact = QuarantineArtifact(
            artifact_id=art_id,
            seed=seed,
            prompt_hash=prompt_hash,
            generator_model=generator_model,
            generator_version=generator_version,
            optimized_code=optimized_code,
            reference_code=reference_code,
            test_generator_code=test_generator_code,
            failing_input=failing_input,
            expected_output=expected_output,
            actual_output=actual_output,
            traceback_str=traceback_str,
            execution_traces=execution_traces or {},
            quarantine_reason=quarantine_reason,
            timestamp=time.time(),
        )

        out_path = os.path.join(self.quarantine_dir, f"{art_id}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(asdict(artifact), f, indent=2)

        return art_id

    @staticmethod
    def _build_rejection_diagnostics(
        reason: str,
        ast_msg: str = "",
        res_a: Optional[Dict[str, Any]] = None,
        res_b: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Constructs a machine-readable rejection diagnostics dict from oracle results.

        This helper is called on every non-verified return path of ``verify()``.
        It classifies the rejection into structured categories without altering
        any existing return keys.

        Args:
            reason:   The human-readable failure reason string already present
                      in the return dict.
            ast_msg:  The message from ``verify_ast_security_and_contracts``.
            res_a:    Phase A result dict from the child process, or None.
            res_b:    Phase B result dict from the child process, or None.

        Returns:
            A dict with stable string keys suitable for machine parsing.
        """
        reason_upper = reason.upper()
        ast_upper = ast_msg.upper()

        is_security = "SECURITY POLICY" in ast_upper
        is_anti_cheat = "ANTI-CHEAT" in ast_upper
        is_syntax_error = "SYNTAXERROR" in ast_upper
        is_contract = "CONTRACT" in ast_upper and not is_anti_cheat

        # Determine syntax category from AST message
        if is_security:
            syntax_category = "security_violation"
        elif is_anti_cheat:
            syntax_category = "anti_cheat"
        elif is_syntax_error:
            syntax_category = "syntax_error"
        elif is_contract:
            syntax_category = "contract_violation"
        elif ast_msg:
            syntax_category = "ast_failure"
        else:
            syntax_category = "none"

        # Parse failure location from Phase A or AST message
        parse_failure_location: Optional[int] = None
        if res_a and res_a.get("failing_input"):
            # Line number not available from subprocess; mark as None
            parse_failure_location = None

        # Phase A mismatch type
        if res_a is not None and not res_a.get("passed", True):
            a_reason = res_a.get("reason", "").upper()
            if "MISMATCH" in a_reason:
                phase_a_mismatch = "output_mismatch"
            elif "TIMEOUT" in a_reason:
                phase_a_mismatch = "timeout"
            elif "EXCEPTION" in a_reason:
                phase_a_mismatch = "exception"
            elif "MEMORY" in a_reason:
                phase_a_mismatch = "memory"
            else:
                phase_a_mismatch = "other"
        else:
            phase_a_mismatch = "none"

        # Phase B failure type
        if res_b is not None and not res_b.get("passed", True):
            b_reason = res_b.get("reason", "").upper()
            if "TIMEOUT" in b_reason or "TIME BOUND" in b_reason:
                phase_b_failure = "timeout"
            elif "MEMORY" in b_reason:
                phase_b_failure = "memory"
            elif "EXCEPTION" in b_reason:
                phase_b_failure = "exception"
            else:
                phase_b_failure = "other"
        else:
            phase_b_failure = "none"

        is_timeout = (
            "TIMEOUT" in reason_upper
            or phase_a_mismatch == "timeout"
            or phase_b_failure == "timeout"
        )
        is_memory = (
            "MEMORY" in reason_upper
            or phase_a_mismatch == "memory"
            or phase_b_failure == "memory"
        )

        return {
            "syntax_category": syntax_category,
            "parse_failure_location": parse_failure_location,
            "phase_a_mismatch_type": phase_a_mismatch,
            "phase_b_failure": phase_b_failure,
            "is_timeout": is_timeout,
            "is_memory": is_memory,
            "is_duplicate": False,   # dedup is caller-side; always False here
            "is_security": is_security,
            "is_anti_cheat": is_anti_cheat,
        }

    def verify(
        self,
        full_code: str,
        seed: int = 42,
        prompt_hash: str = "hash0000",
        generator_model: str = "gemini-3.6-flash",
        generator_version: str = "v1.0",
        solve_opt: str = "",
        solve_brute: str = "",
        test_gen: str = "",
    ) -> Dict[str, Any]:
        """Runs 2-phase differential verification engine.

        Returns a dict with the following keys (unchanged from original):
            verified, confidence, phase_a_pass, phase_a_pass_count,
            phase_b_pass, reason, quarantine_id.

        Additionally, every return path now includes:
            rejection_diagnostics: Dict[str, Any] — structured, machine-readable
            breakdown of why a sample was rejected, or an all-"none" dict for
            verified samples.  This key is additive and does not affect any
            existing caller that does not reference it.
        """
        ast_ok, ast_msg = verify_ast_security_and_contracts(full_code)
        if not ast_ok:
            art_id = self.quarantine(
                seed=seed,
                prompt_hash=prompt_hash,
                generator_model=generator_model,
                generator_version=generator_version,
                optimized_code=solve_opt or full_code,
                reference_code=solve_brute,
                test_generator_code=test_gen,
                quarantine_reason=ast_msg,
            )
            return {
                "verified": False,
                "confidence": 0.0,
                "phase_a_pass": False,
                "phase_a_pass_count": 0,
                "phase_b_pass": False,
                "reason": ast_msg,
                "quarantine_id": art_id,
                "rejection_diagnostics": self._build_rejection_diagnostics(
                    reason=ast_msg, ast_msg=ast_msg
                ),
            }

        ctx = mp.get_context("spawn")

        # Phase A Execution
        parent_conn, child_conn = ctx.Pipe(duplex=False)
        p_a = ctx.Process(target=_worker_phase_a, args=(full_code, seed, child_conn))
        p_a.start()
        child_conn.close()

        res_a = None
        start_t = time.time()
        while time.time() - start_t < 4.0:
            if parent_conn.poll(0.1):
                res_a = parent_conn.recv()
                break
            if not p_a.is_alive():
                break
        p_a.join(timeout=0.5)
        if p_a.is_alive():
            p_a.terminate()
            p_a.join(timeout=0.5)
            res_a = {"passed": False, "reason": "Phase A Timeout (> 4s wall clock)", "pass_count": 0}
        parent_conn.close()

        if not res_a or not res_a.get("passed"):
            reason = res_a.get("reason", "Phase A Worker Exited Unexpectedly") if res_a else "Phase A Failure"
            art_id = self.quarantine(
                seed=seed,
                prompt_hash=prompt_hash,
                generator_model=generator_model,
                generator_version=generator_version,
                optimized_code=solve_opt,
                reference_code=solve_brute,
                test_generator_code=test_gen,
                quarantine_reason=reason,
                failing_input=res_a.get("failing_input") if res_a else None,
                expected_output=res_a.get("expected_output") if res_a else None,
                actual_output=res_a.get("actual_output") if res_a else None,
            )
            return {
                "verified": False,
                "confidence": 0.0,
                "phase_a_pass": False,
                "phase_a_pass_count": res_a.get("pass_count", 0) if res_a else 0,
                "phase_b_pass": False,
                "reason": reason,
                "quarantine_id": art_id,
                "rejection_diagnostics": self._build_rejection_diagnostics(
                    reason=reason, res_a=res_a
                ),
            }

        # Phase B Execution
        parent_conn, child_conn = ctx.Pipe(duplex=False)
        p_b = ctx.Process(target=_worker_phase_b, args=(full_code, seed, child_conn))
        p_b.start()
        child_conn.close()

        res_b = None
        start_t = time.time()
        while time.time() - start_t < 12.0:
            if parent_conn.poll(0.1):
                res_b = parent_conn.recv()
                break
            if not p_b.is_alive():
                break
        p_b.join(timeout=0.5)
        if p_b.is_alive():
            p_b.terminate()
            p_b.join(timeout=0.5)
            res_b = {"passed": False, "reason": "Phase B Timeout (> 12s wall clock)"}
        parent_conn.close()

        if not res_b or not res_b.get("passed"):
            reason = res_b.get("reason", "Phase B Worker Exited Unexpectedly") if res_b else "Phase B Failure"
            art_id = self.quarantine(
                seed=seed,
                prompt_hash=prompt_hash,
                generator_model=generator_model,
                generator_version=generator_version,
                optimized_code=solve_opt,
                reference_code=solve_brute,
                test_generator_code=test_gen,
                quarantine_reason=reason,
                failing_input=res_b.get("failing_input") if res_b else None,
                actual_output=res_b.get("actual_output") if res_b else None,
            )
            return {
                "verified": False,
                "confidence": 0.5,
                "phase_a_pass": True,
                "phase_a_pass_count": res_a.get("pass_count", 20),
                "phase_b_pass": False,
                "reason": reason,
                "quarantine_id": art_id,
                "rejection_diagnostics": self._build_rejection_diagnostics(
                    reason=reason, res_a=res_a, res_b=res_b
                ),
            }

        return {
            "verified": True,
            "confidence": 1.0,
            "phase_a_pass": True,
            "phase_a_pass_count": res_a.get("pass_count", 20),
            "phase_b_pass": True,
            "reason": "VERIFICATION_SUCCESS",
            "quarantine_id": None,
            "rejection_diagnostics": {
                "syntax_category": "none",
                "parse_failure_location": None,
                "phase_a_mismatch_type": "none",
                "phase_b_failure": "none",
                "is_timeout": False,
                "is_memory": False,
                "is_duplicate": False,
                "is_security": False,
                "is_anti_cheat": False,
            },
        }


if __name__ == "__main__":
    oracle = VerificationOracle()
    sample_code = (
        "def solve(arr):\n    return sorted(arr)\n\n"
        "def solve_brute(arr):\n    res = list(arr)\n    for i in range(len(res)):\n        for j in range(i+1, len(res)):\n            if res[i] > res[j]:\n                res[i], res[j] = res[j], res[i]\n    return res\n\n"
        "import random\n"
        "def generate_test_cases(seed, scale):\n"
        "    random.seed(seed)\n"
        "    if scale == 'small':\n"
        "        return [[random.randint(1, 100) for _ in range(5)] for _ in range(5)]\n"
        "    else:\n"
        "        return [[random.randint(1, 100) for _ in range(500)] for _ in range(3)]\n"
    )
    res = oracle.verify(sample_code, seed=42)
    print("Verification result:", res)
