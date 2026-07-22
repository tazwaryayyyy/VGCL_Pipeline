import ast
import copy
import io
import math
import multiprocessing as mp
import os
import socket
import sys
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple

try:
    import resource
except ImportError:
    resource = None

ALLOWED_MODULES = {
    "math",
    "collections",
    "heapq",
    "bisect",
    "itertools",
    "functools",
    "typing",
    "random",
    "re",
    "string",
    "dataclasses",
    "copy",
    "operator",
    "array",
}

FORBIDDEN_BUILTINS = {
    "open",
    "eval",
    "exec",
    "compile",
    "breakpoint",
    "input",
    "exit",
    "quit",
    "__import__",
}

FORBIDDEN_ATTRIBUTES = {
    "__subclasses__",
    "__bases__",
    "__mro__",
    "__globals__",
    "__code__",
    "__closure__",
    "__builtins__",
}


def _set_hardened_sandbox_limits() -> None:
    """Injects POSIX OS-level CPU limits and blocks network access safely."""
    if resource and hasattr(resource, "RLIMIT_CPU"):
        try:
            resource.setrlimit(resource.RLIMIT_CPU, (2, 2))
        except Exception:
            pass

    # Disable socket creation at module level and C-module level
    try:
        import socket
        socket.socket = None
        socket.create_connection = None
    except Exception:
        pass

    try:
        import _socket
        _socket.socket = None
    except Exception:
        pass


def _check_peak_rss_limit() -> Optional[str]:
    """Monitors peak RSS memory usage to prevent memory explosion."""
    if resource and hasattr(resource, "getrusage"):
        try:
            max_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            limit_kb = 512 * 1024  # 512 MB in KB
            if sys.platform == "darwin":
                limit_kb *= 1024  # macOS reports in bytes
            if max_rss > limit_kb:
                return f"Memory Limit Exceeded: Peak RSS ({max_rss}) > 512MB"
        except Exception:
            pass
    return None


def _truncate_repr(obj: Any, max_len: int = 1000) -> str:
    """Safely truncates string representations of failing input payloads to prevent IPC bloat."""
    try:
        r = repr(obj)
        if len(r) > max_len:
            return r[:max_len] + "... [TRUNCATED]"
        return r
    except Exception:
        return "<Unrepresentable Object>"


def _are_results_equal(a: Any, b: Any, rel_tol: float = 1e-6, abs_tol: float = 1e-6) -> bool:
    """Robust differential equality check handling floats, NaNs, nested containers, and numerical types."""
    if type(a) != type(b):
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)
        return False

    if isinstance(a, float):
        if math.isnan(a) and math.isnan(b):
            return True
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)

    if isinstance(a, (list, tuple)):
        if len(a) != len(b):
            return False
        return all(_are_results_equal(x, y, rel_tol, abs_tol) for x, y in zip(a, b))

    if isinstance(a, dict):
        if a.keys() != b.keys():
            return False
        return all(_are_results_equal(a[k], b[k], rel_tol, abs_tol) for k in a)

    return a == b


class SecurityVisitor(ast.NodeVisitor):
    """AST Security visitor enforcing module, builtin, and attribute whitelists."""

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
                self.violations.append(f"Forbidden module import from: '{node.module}'")
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name):
        if node.id in FORBIDDEN_BUILTINS:
            self.violations.append(f"Forbidden builtin access: '{node.id}'")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute):
        if node.attr in FORBIDDEN_ATTRIBUTES:
            self.violations.append(f"Forbidden attribute access: '{node.attr}'")
        self.generic_visit(node)


class ASTAntiCheatVisitor(ast.NodeVisitor):
    """Verifies solve_brute does not invoke solve and checks structural independence."""

    def __init__(self):
        self.calls_solve = False
        self.solve_brute_found = False

    def visit_FunctionDef(self, node: ast.FunctionDef):
        if node.name == "solve_brute":
            self.solve_brute_found = True
            for child in ast.walk(node):
                if isinstance(child, ast.Call):
                    if isinstance(child.func, ast.Name) and child.func.id == "solve":
                        self.calls_solve = True
        self.generic_visit(node)


def verify_ast_anti_cheat(code_str: str) -> Tuple[bool, str]:
    """Inspects AST to enforce security policies and anti-cheat constraints."""
    try:
        tree = ast.parse(code_str)
    except SyntaxError as e:
        return False, f"SyntaxError during AST inspection: {e}"

    sec_visitor = SecurityVisitor()
    sec_visitor.visit(tree)
    if sec_visitor.violations:
        return False, f"Security Policy Violation: {'; '.join(sec_visitor.violations)}"

    visitor = ASTAntiCheatVisitor()
    visitor.visit(tree)

    if not visitor.solve_brute_found:
        return False, "AST Verification Failed: solve_brute() definition not found."

    if visitor.calls_solve:
        return False, "AST Anti-Cheat Triggered: solve_brute() directly calls solve()."

    return True, "AST Anti-Cheat Passed"


def _worker_exec_phase_a(code_str: str, seed: int, conn) -> None:
    """Worker process for Phase A: Small-scale differential verification."""
    _set_hardened_sandbox_limits()
    exec_globals = {}
    current_args = None
    try:
        exec(code_str, exec_globals)
        solve = exec_globals["solve"]
        solve_brute = exec_globals["solve_brute"]
        gen_cases = exec_globals["generate_test_cases"]

        test_cases = gen_cases(seed=seed, scale="small")[:20]

        for idx, args in enumerate(test_cases):
            pass_args = args if isinstance(args, (list, tuple)) else (args,)
            current_args = pass_args

            ans_opt = solve(*copy.deepcopy(pass_args))
            ans_brute = solve_brute(*copy.deepcopy(pass_args))

            if not _are_results_equal(ans_opt, ans_brute):
                conn.send(
                    (
                        False,
                        f"Phase A Mismatch on small test #{idx + 1}: Optimal={ans_opt} vs Brute={ans_brute}",
                        _truncate_repr(pass_args),
                    )
                )
                return

            mem_err = _check_peak_rss_limit()
            if mem_err:
                conn.send((False, f"Phase A {mem_err}", _truncate_repr(current_args)))
                return

        conn.send((True, "Phase A Passed", None))
    except Exception as e:
        tb_lines = traceback.format_exc()
        input_str = _truncate_repr(current_args) if current_args is not None else "N/A"
        conn.send((False, f"Phase A Runtime Exception: {type(e).__name__}: {e}\n{tb_lines}", input_str))


def _worker_exec_phase_bc(code_str: str, seed: int, conn) -> None:
    """Worker process for Phase B (Boundary) & Phase C (Perf/Memory & State Isolation)."""
    _set_hardened_sandbox_limits()
    exec_globals = {}
    current_args = None
    try:
        exec(code_str, exec_globals)
        solve = exec_globals["solve"]
        gen_cases = exec_globals["generate_test_cases"]

        # Phase B: Boundary test assertions
        boundary_cases = gen_cases(seed=seed, scale="small")
        for args in boundary_cases[:3]:
            pass_args = args if isinstance(args, (list, tuple)) else (args,)
            current_args = pass_args

            res_1 = solve(*copy.deepcopy(pass_args))
            res_2 = solve(*copy.deepcopy(pass_args))
            if not _are_results_equal(res_1, res_2):
                conn.send(
                    (
                        False,
                        "Phase B State Mutation Failure: consecutive calls produced different results.",
                        _truncate_repr(pass_args),
                    )
                )
                return

        # Phase C: Performance Verification (N = 10^4 to 10^5) generated inside worker memory space
        large_cases = gen_cases(seed=seed, scale="large")[:5]
        for idx, args in enumerate(large_cases):
            pass_args = args if isinstance(args, (list, tuple)) else (args,)
            current_args = pass_args

            start_t = time.perf_counter()
            _ = solve(*pass_args)
            elapsed = time.perf_counter() - start_t

            if elapsed > 2.0:
                conn.send(
                    (
                        False,
                        f"Phase C Timeout: Large scale test #{idx + 1} took {elapsed:.4f}s (> 2.0s)",
                        _truncate_repr(pass_args),
                    )
                )
                return

            mem_err = _check_peak_rss_limit()
            if mem_err:
                conn.send((False, f"Phase C {mem_err}", _truncate_repr(current_args)))
                return

        conn.send((True, "Phase B & C Passed", None))
    except Exception as e:
        tb_lines = traceback.format_exc()
        input_str = _truncate_repr(current_args) if current_args is not None else "N/A"
        conn.send((False, f"Phase B/C Exception: {type(e).__name__}: {e}\n{tb_lines}", input_str))


def _safe_run_process(ctx: Any, target_func: Any, code_str: str, seed: int, timeout: float) -> Tuple[bool, str, Optional[str]]:
    """Runs target_func in child process with non-blocking IPC polling to eliminate Pipe deadlocks."""
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    p = ctx.Process(target=target_func, args=(code_str, seed, child_conn))
    p.start()
    child_conn.close()

    result = None
    start_time = time.time()
    try:
        while time.time() - start_time < timeout:
            if parent_conn.poll(0.1):
                result = parent_conn.recv()
                break
            if not p.is_alive():
                break

        p.join(timeout=max(0.1, timeout - (time.time() - start_time)))
        if p.is_alive():
            p.terminate()
            p.join(timeout=1.0)
            return False, f"Execution timed out (> {timeout:.1f}s wall clock)", None

        if result is not None:
            return result
        return False, "Process exited unexpectedly without returning a result.", None
    finally:
        parent_conn.close()
        if p.is_alive():
            try:
                p.kill()
                p.join()
            except Exception:
                pass


def run_full_verification(code_str: str, seed: int = 42) -> Tuple[bool, str, Optional[str]]:
    """Runs full 4-tier differential verification engine with non-blocking IPC and security checks."""
    ast_ok, ast_msg = verify_ast_anti_cheat(code_str)
    if not ast_ok:
        return False, ast_msg, None

    ctx = mp.get_context("spawn")

    # Phase A execution
    ok_a, msg_a, input_a = _safe_run_process(ctx, _worker_exec_phase_a, code_str, seed, timeout=4.0)
    if not ok_a:
        return False, msg_a, input_a

    # Phase B & C execution
    ok_bc, msg_bc, input_bc = _safe_run_process(ctx, _worker_exec_phase_bc, code_str, seed, timeout=12.0)
    if not ok_bc:
        return False, msg_bc, input_bc

    return True, "VERIFICATION_SUCCESS", None
