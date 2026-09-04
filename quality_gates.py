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


def extract_declared_constraints(spec: str) -> Dict[str, Any]:
    """Extracts machine-readable scale and resource constraints from problem specification."""
    import re
    text = spec or ""

    # 1. Parse max_n / dimension constraints
    max_n: Optional[int] = None

    # Check scientific notation / powers: 10^5, 10**5, 2*10^5, 2e5
    power_matches = re.findall(r"(?:(?:(\d+)\s*\*\s*)?10\^(\d+)|(?:(\d+)\s*\*\s*)?10\*\*(\d+)|(\d+)e(\d+))", text, re.IGNORECASE)
    for m in power_matches:
        if m[1]:  # 10^k or c*10^k
            coeff = int(m[0]) if m[0] else 1
            val = coeff * (10 ** int(m[1]))
            max_n = max(max_n or 0, val)
        elif m[3]:  # 10**k or c*10**k
            coeff = int(m[2]) if m[2] else 1
            val = coeff * (10 ** int(m[3]))
            max_n = max(max_n or 0, val)
        elif m[4] and m[5]:  # ceK
            val = int(float(f"{m[4]}e{m[5]}"))
            max_n = max(max_n or 0, val)

    # Check direct bounds like N <= 100000, length <= 500, etc.
    bound_patterns = [
        r"(?:N|n|length|size|elements|items|vertices|nodes|nums|array|string)\s*(?:<=|\\le|<|up to|at most|is at most|bounded by)\s*([0-9_]+(?:\.[0-9]+)?)",
        r"(?:<=|\\le|<)\s*([0-9_]+(?:\.[0-9]+)?)\s*(?:elements|items|length|size|operations|N|n)",
        r"(?:max_n|maxn|max_dimension)\s*[:=]\s*([0-9_]+)",
        r"1\s*(?:<=|\\le|<)\s*(?:N|n|length|size)\s*(?:<=|\\le|<)\s*([0-9_]+)",
    ]
    for pat in bound_patterns:
        for match in re.finditer(pat, text, re.IGNORECASE):
            raw_val = match.group(1).replace("_", "")
            try:
                num = int(float(raw_val))
                if num > 0:
                    max_n = max(max_n or 0, num)
            except ValueError:
                continue

    if max_n is None:
        max_n = 100000  # Conservative standard competitive programming default

    # 2. Parse time limit
    time_limit_s = 2.0
    time_match = re.search(r"time\s*limit\s*[:=]?\s*([0-9.]+)\s*(?:s|sec|seconds)?", text, re.IGNORECASE)
    if time_match:
        try:
            time_limit_s = float(time_match.group(1))
        except ValueError:
            pass

    # 3. Parse memory limit
    memory_limit_mb = 512.0
    mem_match = re.search(r"memory\s*limit\s*[:=]?\s*([0-9.]+)\s*(?:mb|megabytes)?", text, re.IGNORECASE)
    if mem_match:
        try:
            memory_limit_mb = float(mem_match.group(1))
        except ValueError:
            pass

    return {
        "max_n": max_n,
        "time_limit_s": time_limit_s,
        "memory_limit_mb": memory_limit_mb,
        "summary": f"N <= {max_n}, time <= {time_limit_s}s, memory <= {memory_limit_mb}MB",
    }


class _ConstraintVisitor(ast.NodeVisitor):
    def __init__(self, dominant_args: set[str]):
        self.dominant_args = set(dominant_args)
        self.flags: List[str] = []
        self._loop_stack: List[ast.AST] = []
        self._memoized = False
        self.recursive_call_count = 0
        self.function_name = ""

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.function_name = node.name
        # Check decorators for memoization (e.g. @lru_cache, @cache)
        for dec in node.decorator_list:
            if isinstance(dec, ast.Name) and dec.id in ("cache", "lru_cache"):
                self._memoized = True
            elif isinstance(dec, ast.Call) and isinstance(dec.func, ast.Name) and dec.func.id in ("cache", "lru_cache"):
                self._memoized = True
        self.generic_visit(node)

    def _is_small_constant_range(self, node: ast.AST) -> bool:
        """Determines if a loop is over a small constant range (<= 32, e.g. alphabet or small bits)."""
        if isinstance(node, ast.Call):
            func_name = ""
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
            if func_name == "range":
                if len(node.args) == 1 and isinstance(node.args[0], ast.Constant):
                    if isinstance(node.args[0].value, (int, float)) and node.args[0].value <= 32:
                        return True
                elif len(node.args) == 2 and isinstance(node.args[0], ast.Constant) and isinstance(node.args[1], ast.Constant):
                    if isinstance(node.args[0].value, (int, float)) and isinstance(node.args[1].value, (int, float)):
                        if abs(node.args[1].value - node.args[0].value) <= 32:
                            return True
        elif isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            if len(node.elts) <= 32 and all(isinstance(e, ast.Constant) for e in node.elts):
                return True
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and len(node.value) <= 32:
            return True
        return False

    def visit_For(self, node: ast.For) -> None:
        is_small = self._is_small_constant_range(node.iter)
        if self._loop_stack and not is_small:
            # Check if any parent loop is also not a small constant
            if any(not self._is_small_constant_range(p.iter if isinstance(p, ast.For) else p) for p in self._loop_stack):
                self.flags.append("nested_loop_dominant_dimension")
        self._loop_stack.append(node)
        self.generic_visit(node)
        self._loop_stack.pop()

    def visit_While(self, node: ast.While) -> None:
        if self._loop_stack:
            self.flags.append("nested_while_loop")
        self._loop_stack.append(node)
        self.generic_visit(node)
        self._loop_stack.pop()

    def visit_Compare(self, node: ast.Compare) -> None:
        # Check for `item in list` inside loop over dominant dimension
        if self._loop_stack:
            for op, comparator in zip(node.ops, node.comparators):
                if isinstance(op, (ast.In, ast.NotIn)):
                    if isinstance(comparator, ast.Name) and comparator.id in self.dominant_args:
                        self.flags.append("nested_linear_scan_in_loop")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name):
            if node.func.id == self.function_name and self.function_name:
                self.recursive_call_count += 1
        elif isinstance(node.func, ast.Attribute):
            # Check for list.count() or list.index() inside loop
            if node.func.attr in ("count", "index") and self._loop_stack:
                if isinstance(node.func.value, ast.Name) and node.func.value.id in self.dominant_args:
                    self.flags.append("nested_linear_scan_in_loop")
        self.generic_visit(node)


def check_constraint_regime_compatibility(spec: str, solve_opt: str) -> Dict[str, Any]:
    """Deterministically validates whether optimized solution structure matches declared constraint regime."""
    constraints = extract_declared_constraints(spec)
    max_n = constraints.get("max_n", 100000)

    try:
        tree = ast.parse(solve_opt)
    except SyntaxError:
        return {
            "status": "REJECT",
            "declared_constraints": constraints,
            "structural_flags": ["syntax_error"],
            "reason": "optimized solution is not parseable",
        }

    # Find solve function arguments
    solve_func: Optional[ast.FunctionDef] = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "solve":
            solve_func = node
            break

    dominant_args = set()
    if solve_func:
        for arg in solve_func.args.args:
            dominant_args.add(arg.arg)

    visitor = _ConstraintVisitor(dominant_args)
    visitor.visit(tree)

    # Check for exponential recursion without memoization
    if visitor.recursive_call_count >= 2 and not visitor._memoized:
        visitor.flags.append("unmemoized_branching_recursion")

    flags = list(dict.fromkeys(visitor.flags))  # Deduplicate preserving order

    # Decision logic based on declared constraint regime
    if max_n <= 500:
        # Small constraint regime (e.g. N <= 500): O(N^2) or nested iteration is fully compatible and intended.
        return {
            "status": "PASS",
            "declared_constraints": constraints,
            "structural_flags": flags,
            "reason": f"quadratic/nested iteration structure compatible with declared small constraint regime (N<={max_n})",
        }

    # Large constraint regime (N >= 5000)
    if "unmemoized_branching_recursion" in flags:
        return {
            "status": "REJECT",
            "declared_constraints": constraints,
            "structural_flags": flags,
            "reason": f"structural scaling incompatibility: unmemoized branching recursion detected for large constraint regime ({constraints['summary']})",
        }

    if "nested_loop_dominant_dimension" in flags or "nested_linear_scan_in_loop" in flags:
        if max_n >= 5000:
            return {
                "status": "REJECT",
                "declared_constraints": constraints,
                "structural_flags": flags,
                "reason": f"structural scaling incompatibility: obvious O(N^2) nested iteration detected over dominant input dimension for large constraint regime ({constraints['summary']})",
            }

    return {
        "status": "PASS",
        "declared_constraints": constraints,
        "structural_flags": flags,
        "reason": f"structural checks compatible with declared constraint regime ({constraints['summary']})",
    }


# ---------------------------------------------------------------------------
# V7: Authoritative Primary Eligibility Predicate
# ---------------------------------------------------------------------------


def check_primary_eligibility(
    entry: Any,
    condition_type: str = "TARGETED",
    primary_model: str = "gemini-3.6-flash",
) -> Dict[str, Any]:
    """
    Canonical single source of truth for primary dataset eligibility in VGCL Phase 0.

    Distinguishes:
      - PRIMARY_ELIGIBLE: Meets all strict scientific validity, provenance, and control criteria.
      - AUXILIARY_NON_PRIMARY: Verified code package but ineligible for primary dataset
        (e.g., fallback generator used, or target was unverified / did not replicate on base model).
      - QUARANTINED / REJECTED: Failed oracle verification, deduplication, or quality gates.

    Args:
        entry: LedgerEntry object or dictionary containing candidate record metadata.
        condition_type: "TARGETED" for targeted curriculum or "UNIFORM" for uniform control.
        primary_model: Expected frozen primary generator model identifier.

    Returns:
        Dict with keys:
          - "status": "PRIMARY_ELIGIBLE" | "AUXILIARY_NON_PRIMARY" | "REJECTED"
          - "is_primary": bool
          - "reasons": List[str]
          - "condition_type": str
    """
    import json as _json

    # Canonical subtypes and cluster compatibility imported lazily to prevent circular imports
    canonical_subtypes = [
        "Type A: sliding-window extrema",
        "Type B: bounded-transition DP",
        "Type C: prefix-sum deque optimization",
        "Type D: convex optimization deque",
        "Type E: shortest-path deque variants",
        "Type F: hybrid stack/deque structures",
    ]
    canonical_cluster_compat = {
        "FC_01_MICRO_CORRECTNESS": [
            "Type A: sliding-window extrema",
            "Type B: bounded-transition DP",
            "Type D: convex optimization deque",
            "Type F: hybrid stack/deque structures",
        ],
        "FC_02_SCALE_DIVERGENCE": [
            "Type B: bounded-transition DP",
            "Type C: prefix-sum deque optimization",
            "Type D: convex optimization deque",
        ],
        "FC_03_TERMINATION": [
            "Type C: prefix-sum deque optimization",
            "Type E: shortest-path deque variants",
            "Type F: hybrid stack/deque structures",
        ],
        "FC_04_BOUNDS_VIOLATION": [
            "Type A: sliding-window extrema",
            "Type D: convex optimization deque",
            "Type E: shortest-path deque variants",
        ],
        "FC_05_RESOURCE_FAILURE": [
            "Type C: prefix-sum deque optimization",
            "Type D: convex optimization deque",
            "Type E: shortest-path deque variants",
        ],
        "FC_06_FORMAT_CONTRACT": [
            "Type A: sliding-window extrema",
            "Type F: hybrid stack/deque structures",
        ],
        "FC_07_STATE_INCONSISTENCY": [
            "Type B: bounded-transition DP",
            "Type E: shortest-path deque variants",
            "Type F: hybrid stack/deque structures",
        ],
        "FC_08_UNKNOWN": list(canonical_subtypes),
    }

    if isinstance(entry, dict):
        d = entry
    elif hasattr(entry, "to_dict"):
        d = entry.to_dict()
    else:
        d = vars(entry)

    reasons: List[str] = []
    condition = condition_type.upper()

    # 1. Oracle Verification Invariant (Phase A + Phase B + Approved status)
    phase_a_count = int(d.get("phase_a_pass_count", 0))
    phase_b_pass = bool(d.get("phase_b_pass", False))
    ver_status = str(d.get("verification_status", ""))
    if phase_a_count <= 0 or not phase_b_pass or ver_status != "APPROVED":
        reasons.append(
            f"Oracle verification not satisfied (Phase A: {phase_a_count}, Phase B: {phase_b_pass}, Status: {ver_status})"
        )

    # 2. Structural Deduplication Invariant
    ast_hash = str(d.get("ast_hash", ""))
    sim_score = float(d.get("similarity_score", 0.0))
    if not ast_hash or sim_score >= 0.70:
        reasons.append(
            f"Deduplication invariant not satisfied (ast_hash='{ast_hash}', sim_score={sim_score:.3f} >= 0.70)"
        )

    # 3. Evaluation Leakage & Reviewer Invariant
    if bool(d.get("reviewer_override", False)):
        reasons.append("Reviewer override detected (manual intervention violates unassisted generation)")

    # 4. Generator Model & Fallback Invariant (No fallback contamination in primary)
    fallback_used = bool(d.get("fallback_used", False))
    fb_p1 = bool(d.get("fallback_used_pass1", False))
    fb_p2 = bool(d.get("fallback_used_pass2", False))
    gen_model = str(d.get("generator_model", ""))
    m_p1 = str(d.get("generator_model_pass1", gen_model))
    m_p2 = str(d.get("generator_model_pass2", gen_model))
    gen_policy = str(d.get("generator_policy", ""))

    if fallback_used or fb_p1 or fb_p2:
        reasons.append("Generator fallback activated (fallback output cannot enter primary dataset)")
    if gen_model != primary_model or m_p1 != primary_model or m_p2 != primary_model:
        reasons.append(
            f"Generator model mismatch (expected '{primary_model}', observed '{gen_model}' [p1='{m_p1}', p2='{m_p2}'])"
        )
    if gen_policy != "EXPERIMENTAL_GENERATOR_POLICY":
        reasons.append(f"Invalid generator policy: '{gen_policy}'")

    # 5. Shared Domain Support Invariant
    declared_subtype = str(d.get("monotonic_queue_subtype", ""))
    if declared_subtype not in canonical_subtypes:
        reasons.append(f"Subtype '{declared_subtype}' outside declared common domain support")

    # 6. Condition-Specific Provenance and Targeting Invariants
    probe_hash = str(d.get("probe_result_hash", ""))

    if condition == "TARGETED":
        val_status = str(d.get("target_validation_status", ""))
        target_cluster = str(d.get("target_failure_cluster", ""))
        target_evidence_raw = d.get("target_evidence", "{}")

        if val_status != "VALIDATED":
            reasons.append(f"Target validation status is '{val_status}' (must be 'VALIDATED' for targeted primary)")
        if not probe_hash:
            reasons.append("Missing probe_result_hash (empirical probe provenance required)")

        target_evidence: Dict[str, Any] = {}
        if isinstance(target_evidence_raw, str):
            try:
                target_evidence = _json.loads(target_evidence_raw)
            except Exception:
                target_evidence = {}
        elif isinstance(target_evidence_raw, dict):
            target_evidence = target_evidence_raw

        if not target_evidence or "exception_type" not in target_evidence:
            reasons.append("Missing or unparseable target_evidence in probe record")

        allowed_subtypes = canonical_cluster_compat.get(target_cluster, [])
        if declared_subtype not in allowed_subtypes:
            reasons.append(
                f"Subtype '{declared_subtype}' is incompatible with target cluster '{target_cluster}' under CLUSTER_SUBTYPE_COMPATIBILITY"
            )

    elif condition == "UNIFORM":
        target_cluster = str(d.get("target_failure_cluster", ""))
        val_status = str(d.get("target_validation_status", ""))
        if target_cluster != "UNIFORM_CONTROL":
            reasons.append(f"Uniform control item must have target_failure_cluster='UNIFORM_CONTROL', observed '{target_cluster}'")
        if val_status != "NOT_APPLICABLE":
            reasons.append(f"Uniform control item must have target_validation_status='NOT_APPLICABLE', observed '{val_status}'")
        if not probe_hash:
            reasons.append("Missing probe_result_hash for base model difficulty measurement")
    else:
        raise ValueError(f"Unknown condition_type: '{condition_type}'. Must be 'TARGETED' or 'UNIFORM'.")

    is_primary = len(reasons) == 0
    if is_primary:
        status = "PRIMARY_ELIGIBLE"
    elif ver_status == "APPROVED" and (fallback_used or d.get("target_validation_status") == "UNVERIFIED"):
        status = "AUXILIARY_NON_PRIMARY"
    else:
        status = "REJECTED"

    return {
        "status": status,
        "is_primary": is_primary,
        "reasons": reasons,
        "condition_type": condition,
    }


def check_primary_targeted_eligibility(entry: Any, primary_model: str = "gemini-3.6-flash") -> Dict[str, Any]:
    """Convenience wrapper for targeted primary eligibility check."""
    return check_primary_eligibility(entry, condition_type="TARGETED", primary_model=primary_model)


def check_primary_uniform_eligibility(entry: Any, primary_model: str = "gemini-3.6-flash") -> Dict[str, Any]:
    """Convenience wrapper for uniform control primary eligibility check."""
    return check_primary_eligibility(entry, condition_type="UNIFORM", primary_model=primary_model)
