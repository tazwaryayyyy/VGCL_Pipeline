"""Deterministic scientific quality gates for VGCL Phase 0.

These helpers are deliberately small and auditable. They do not judge problem
quality, mutate generated algorithms, or replace oracle verification.
"""

import ast
import copy
from typing import Any, Dict, List, Optional, Tuple

from cfg_dedup import compute_ast_hashes, compute_structural_similarity


REFERENCE_INDEPENDENCE_THRESHOLD = 0.85


def _function_source(code: str, name: str, canonical_name: str) -> Optional[str]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            cloned = copy.deepcopy(node)
            cloned.name = canonical_name
            return ast.unparse(ast.Module(body=[cloned], type_ignores=[]))
    return None


def check_reference_independence(
    solve_opt: str,
    solve_brute: str,
    threshold: float = REFERENCE_INDEPENDENCE_THRESHOLD,
) -> Dict[str, Any]:
    """Checks whether optimized/reference implementations are near-isomorphic.

    The method reuses the repository's normalized AST token similarity but
    applies it only to the two solution functions. The threshold is stricter
    than corpus deduplication (0.70) because two functions solving the same
    problem will naturally share signatures and domain vocabulary.
    """
    opt_src = _function_source(solve_opt, "solve", "candidate_solution")
    brute_src = _function_source(solve_brute, "solve_brute", "candidate_solution")
    if not opt_src or not brute_src:
        return {
            "status": "REJECT",
            "similarity_score": 1.0,
            "threshold": threshold,
            "method": "normalized_ast_ngram_similarity_function_only",
            "reason": "required solution function missing",
        }

    opt_ast_hash, opt_cfg_hash, opt_tokens = compute_ast_hashes(opt_src)
    brute_ast_hash, brute_cfg_hash, brute_tokens = compute_ast_hashes(brute_src)
    score = compute_structural_similarity(opt_tokens, brute_tokens)
    identical_shape = opt_ast_hash == brute_ast_hash and opt_cfg_hash == brute_cfg_hash
    rejected = identical_shape or score >= threshold
    return {
        "status": "REJECT" if rejected else "PASS",
        "similarity_score": round(score, 6),
        "threshold": threshold,
        "method": "normalized_ast_ngram_similarity_function_only",
        "optimized_ast_hash": opt_ast_hash,
        "optimized_cfg_hash": opt_cfg_hash,
        "reference_ast_hash": brute_ast_hash,
        "reference_cfg_hash": brute_cfg_hash,
        "reason": "near-isomorphic optimized/reference structure" if rejected else "structurally distinct",
    }


class _FeatureVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.calls: set[str] = set()
        self.names: set[str] = set()
        self.attrs: set[str] = set()
        self.max_loop_depth = 0
        self._loop_depth = 0
        self.has_compare_in_loop = False
        self.has_recursion = False
        self.current_function = ""

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        prev = self.current_function
        self.current_function = node.name
        self.generic_visit(node)
        self.current_function = prev

    def visit_Name(self, node: ast.Name) -> None:
        self.names.add(node.id.lower())

    def visit_Attribute(self, node: ast.Attribute) -> None:
        self.attrs.add(node.attr.lower())
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name):
            name = node.func.id.lower()
            self.calls.add(name)
            if name == self.current_function:
                self.has_recursion = True
        elif isinstance(node.func, ast.Attribute):
            self.calls.add(node.func.attr.lower())
        self.generic_visit(node)

    def _visit_loop(self, node: ast.AST) -> None:
        self._loop_depth += 1
        self.max_loop_depth = max(self.max_loop_depth, self._loop_depth)
        self.generic_visit(node)
        self._loop_depth -= 1

    def visit_For(self, node: ast.For) -> None:
        self._visit_loop(node)

    def visit_While(self, node: ast.While) -> None:
        self._visit_loop(node)

    def visit_Compare(self, node: ast.Compare) -> None:
        if self._loop_depth:
            self.has_compare_in_loop = True
        self.generic_visit(node)


def _features(code: str) -> Optional[_FeatureVisitor]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    visitor = _FeatureVisitor()
    visitor.visit(tree)
    return visitor


def _has_any(text: str, words: Tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(word in lowered for word in words)


def validate_subtype(subtype: str, spec: str, solve_opt: str) -> Dict[str, Any]:
    """Applies weak deterministic evidence checks for declared VGCL subtypes."""
    f = _features(solve_opt)
    if f is None:
        return {"status": "REJECT", "subtype": subtype, "checks": [], "reason": "optimized solution is not parseable"}

    names = f.names | f.calls | f.attrs
    text = f"{spec}\n{solve_opt}".lower()
    checks: List[str] = []

    def require(label: str, ok: bool) -> bool:
        checks.append(f"{label}:{'pass' if ok else 'fail'}")
        return ok

    has_deque = "deque" in names
    has_stack = "stack" in names or {"append", "pop"}.issubset(names)
    has_dp = "dp" in names or "memo" in names or _has_any(text, ("dynamic programming", "transition"))
    has_prefix = "prefix" in names or _has_any(text, ("prefix sum", "prefix-sum"))
    has_graph = _has_any(text, ("graph", "edge", "shortest path", "bfs")) or {"adj", "graph"}.intersection(names)
    has_hull = _has_any(text, ("convex", "hull", "slope", "line", "cht")) or {"hull", "slope", "line"}.intersection(names)

    if subtype.startswith("Type A:"):
        ok = (
            require("window/extrema evidence", _has_any(text, ("window", "maximum", "minimum", "extrema")))
            and require("deque-like state", has_deque)
            and require("monotonic comparison loop", f.has_compare_in_loop)
        )
    elif subtype.startswith("Type B:"):
        ok = (
            require("dp/transition evidence", has_dp)
            and require("bounded iteration structure", f.max_loop_depth >= 2 or has_deque)
        )
    elif subtype.startswith("Type C:"):
        ok = (
            require("prefix evidence", has_prefix)
            and require("deque-like state", has_deque)
            and require("comparison loop", f.has_compare_in_loop)
        )
    elif subtype.startswith("Type D:"):
        ok = (
            require("convex/hull evidence", bool(has_hull))
            and require("candidate queue/deque state", has_deque or "hull" in names)
        )
    elif subtype.startswith("Type E:"):
        ok = (
            require("graph/shortest-path evidence", bool(has_graph))
            and require("deque-like state", has_deque)
        )
    elif subtype.startswith("Type F:"):
        ok = (
            require("stack evidence", bool(has_stack))
            and require("deque evidence", has_deque)
        )
    else:
        return {
            "status": "NOT_VALIDATABLE",
            "subtype": subtype,
            "checks": checks,
            "reason": "unknown subtype",
        }

    return {
        "status": "PASS" if ok else "REJECT",
        "subtype": subtype,
        "checks": checks,
        "reason": "deterministic subtype evidence satisfied" if ok else "declared subtype lacks deterministic structural evidence",
    }


def derive_phase_a_scale_policy(solve_brute: str) -> Dict[str, Any]:
    """Derives a conservative Phase A dimension cap from brute-force structure."""
    brute_src = _function_source(solve_brute, "solve_brute", "solve_brute") or solve_brute
    f = _features(brute_src)
    if f is None:
        return {
            "policy": "parse_unavailable_default",
            "maximum_dimension": 8,
            "rationale": "brute parser unavailable; preserve existing small-input cap",
        }

    combinatorial_calls = {"permutations", "combinations", "product", "combinations_with_replacement"}
    pathological = f.has_recursion or f.max_loop_depth >= 3 or bool(f.calls & combinatorial_calls)
    if pathological:
        return {
            "policy": "pathological_bruteforce_cap",
            "maximum_dimension": 5,
            "rationale": "recursive/combinatorial/deep-loop brute structure detected; keep Phase A small enough for reference execution",
        }
    return {
        "policy": "default_small_scale",
        "maximum_dimension": 8,
        "rationale": "no recursive/combinatorial/deep-loop brute structure detected; preserve existing N<=8 semantics",
    }


def _max_dimension(value: Any) -> int:
    if isinstance(value, (str, bytes)):
        return len(value)
    if isinstance(value, dict):
        dims = [len(value)]
        for key, val in value.items():
            dims.append(_max_dimension(key))
            dims.append(_max_dimension(val))
        return max(dims) if dims else 0
    if isinstance(value, (list, tuple, set, frozenset)):
        dims = [len(value)]
        dims.extend(_max_dimension(item) for item in value)
        return max(dims) if dims else 0
    if isinstance(value, int) and value >= 0:
        return value
    return 0


def case_within_phase_a_policy(args: Any, policy: Dict[str, Any]) -> bool:
    limit = int(policy.get("maximum_dimension", 8))
    return _max_dimension(args) <= limit
