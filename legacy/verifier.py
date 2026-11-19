import ast
import copy
import io
import json
import math
import os
import sys
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple

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




def _truncate_repr(obj: Any, max_len: int = 500) -> str:
    """Safely truncates string representations of payloads to prevent IPC Pipe buffer overflow."""
    try:
        r = repr(obj)
        if len(r) > max_len:
            return r[:max_len] + "... [TRUNCATED]"
        return r
    except Exception:
        return "<Unrepresentable Object>"


def _are_results_equal(a: Any, b: Any, rel_tol: float = 1e-6, abs_tol: float = 1e-6) -> bool:
    """Type-aware differential equality check with NumPy scalar unwrapping and sequence coercion."""
    # Step 1: Unwrap NumPy scalars (np.int64, np.float64, etc.) to native Python types.
    if hasattr(a, "item"):
        a = a.item()
    if hasattr(b, "item"):
        b = b.item()

    # Step 2: Sequence equivalence — list vs tuple are considered equivalent containers.
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return False
        return all(_are_results_equal(x, y, rel_tol, abs_tol) for x, y in zip(a, b))

    # Step 3: Strict primitive type guard.
    if type(a) is not type(b):
        # Allow mixed int/float numerics, but never conflate bool with int (e.g. 1 vs True).
        if isinstance(a, bool) or isinstance(b, bool):
            return False
        if not (isinstance(a, (int, float)) and isinstance(b, (int, float))):
            return False

    # Step 4: Float & NaN handling.
    if isinstance(a, float) or isinstance(b, float):
        if math.isnan(a) and math.isnan(b):
            return True
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)

    if isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            return False
        return all(_are_results_equal(a[k], b[k], rel_tol, abs_tol) for k in a)

    # Step 5: Exact match fallback.
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





def run_full_verification(code_str: str, seed: int = 42) -> Tuple[bool, str, Optional[str]]:
    """Runs full 4-tier differential verification engine with non-blocking IPC and security checks."""
    ast_ok, ast_msg = verify_ast_anti_cheat(code_str)
    if not ast_ok:
        return False, ast_msg, None

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    try:
        from isolated_executor import verify_legacy_phase_isolated
    except ImportError:
        from ..isolated_executor import verify_legacy_phase_isolated

    ok_a, msg_a, input_a = verify_legacy_phase_isolated("a", code_str, seed, timeout_s=4.0)
    if not ok_a:
        return False, msg_a, input_a

    ok_bc, msg_bc, input_bc = verify_legacy_phase_isolated("bc", code_str, seed, timeout_s=12.0)
    if not ok_bc:
        return False, msg_bc, input_bc

    return True, "VERIFICATION_SUCCESS", None


def evaluate_and_route(raw_candidate_str: str) -> Dict[str, Any]:
    """Routes raw JSON strings or candidates through full verification for the salvage pipeline."""
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    try:
        from isolated_executor import SandboxInfrastructureError
    except ImportError:
        from ..isolated_executor import SandboxInfrastructureError
    try:
        # Guard: if the string is a raw LLM generation failure message, quarantine it immediately
        # without attempting JSON deserialization (which would raise a split/parse error).
        if isinstance(raw_candidate_str, str) and raw_candidate_str.startswith("LLM Generation Exception:"):
            return {
                "status": "QUARANTINE",
                "data": {"raw": raw_candidate_str},
                "reason": raw_candidate_str,
                "failing_input": None,
            }

        data = json.loads(raw_candidate_str) if isinstance(raw_candidate_str, str) else raw_candidate_str
        code = data.get("code", "")
        if not code and "solve_opt" in data:
            code = f"{data.get('solve_opt', '')}\n\n{data.get('solve_brute', '')}\n\n{data.get('test_gen', '')}"

        seed = data.get("seed", 42)
        verified, ver_msg, failing_input = run_full_verification(code, seed=seed)

        if verified:
            return {"status": "APPROVED", "data": data, "message": ver_msg}
        elif "AST Verification Failed" in ver_msg or "Security Policy Violation" in ver_msg:
            return {"status": "QUARANTINE", "data": data, "reason": ver_msg, "failing_input": failing_input}
        else:
            return {"status": "REVIEW_QUEUE", "data": data, "reason": ver_msg, "failing_input": failing_input}
    except SandboxInfrastructureError:
        raise
    except Exception as e:
        return {"status": "QUARANTINE", "data": {"raw": raw_candidate_str}, "reason": f"ParseException: {e}", "failing_input": None}
