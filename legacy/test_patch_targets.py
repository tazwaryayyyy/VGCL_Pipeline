"""
Unit tests for the three critical patches:
  Target 1 - _are_results_equal: NumPy scalars, list/tuple coercion, bool/int guard.
  Target 2 - ASTCanonicalizer: Closure scope inheritance in visit_FunctionDef.
  Target 3 - evaluate_and_route: LLM exception string quarantine without split error.
"""
import sys
import math


# ---------------------------------------------------------------------------
# Target 1: _are_results_equal
# ---------------------------------------------------------------------------
def test_are_results_equal():
    print("--- Target 1: _are_results_equal Patch ---")
    from verifier import _are_results_equal

    # (a) list vs tuple must be equivalent
    assert _are_results_equal([1, 2], (1, 2)), \
        "FAIL: [1, 2] vs (1, 2) should be True (sequence coercion)"
    print("  [PASS] _are_results_equal([1, 2], (1, 2)) == True")

    # (b) int 1 vs bool True must NOT be equal
    assert not _are_results_equal(1, True), \
        "FAIL: 1 vs True should be False (bool guard)"
    print("  [PASS] _are_results_equal(1, True) == False")

    # (c) np.int64(5) vs 5 must be equal
    try:
        import numpy as np
        assert _are_results_equal(np.int64(5), 5), \
            "FAIL: np.int64(5) vs 5 should be True (NumPy scalar unwrap)"
        print("  [PASS] _are_results_equal(np.int64(5), 5) == True")

        # bonus: np.float64 close check
        assert _are_results_equal(np.float64(1.0), 1.0), \
            "FAIL: np.float64(1.0) vs 1.0 should be True"
        print("  [PASS] _are_results_equal(np.float64(1.0), 1.0) == True")
    except ImportError:
        print("  [SKIP] NumPy not installed -- skipping np.int64 test")

    # (d) nested list with NaN
    assert _are_results_equal([float("nan"), 1.0], [float("nan"), 1.0]), \
        "FAIL: nested NaN list should be True"
    print("  [PASS] _are_results_equal([nan, 1.0], [nan, 1.0]) == True")

    # (e) False vs 0 must not be equal (bool guard symmetric)
    assert not _are_results_equal(False, 0), \
        "FAIL: False vs 0 should be False (bool guard)"
    print("  [PASS] _are_results_equal(False, 0) == False")

    # (f) different-length list vs tuple
    assert not _are_results_equal([1, 2, 3], (1, 2)), \
        "FAIL: different-length sequences should be False"
    print("  [PASS] _are_results_equal([1,2,3], (1,2)) == False (length mismatch)")

    # (g) int vs float mixed is allowed
    assert _are_results_equal(1, 1.0), \
        "FAIL: int 1 vs float 1.0 should be True (mixed numeric)"
    print("  [PASS] _are_results_equal(1, 1.0) == True (mixed int/float)")

    print("  => All Target 1 checks PASSED\n")


# ---------------------------------------------------------------------------
# Target 2: ASTCanonicalizer nested closure scope inheritance
# ---------------------------------------------------------------------------
def test_ast_canonicalizer_closure_scope():
    print("--- Target 2: ASTCanonicalizer Closure Scope Inheritance ---")
    from dedup import get_canonical_tokens

    # Code with a nested helper that closes over a parent-scope variable.
    code_a = """
def solve(graph, start):
    visited = set()
    def dfs(node):
        if node in visited:
            return
        visited.add(node)
        for neighbor in graph[node]:
            dfs(neighbor)
    dfs(start)
    return visited
"""
    # Semantically identical code with renamed variables -- canonicalization should
    # produce the same token stream both times (idempotent) and across renames.
    code_b = """
def solve(g, s):
    seen = set()
    def dfs(n):
        if n in seen:
            return
        seen.add(n)
        for nb in g[n]:
            dfs(nb)
    dfs(s)
    return seen
"""
    tokens_a1 = get_canonical_tokens(code_a)
    tokens_a2 = get_canonical_tokens(code_a)  # Must be idempotent.
    tokens_b  = get_canonical_tokens(code_b)

    assert tokens_a1 == tokens_a2, \
        "FAIL: get_canonical_tokens is not idempotent for closure code"
    print("  [PASS] Idempotent: same code produces same tokens on repeated calls")

    assert tokens_a1 == tokens_b, (
        "FAIL: Semantically identical closure code should yield same canonical tokens.\n"
        f"  tokens_a1[:20] = {tokens_a1[:20]}\n"
        f"  tokens_b[:20]  = {tokens_b[:20]}"
    )
    print("  [PASS] Closure scope inheritance: renamed variables canonicalize identically")
    print("  => All Target 2 checks PASSED\n")


# ---------------------------------------------------------------------------
# Target 3: evaluate_and_route LLM exception string quarantine
# ---------------------------------------------------------------------------
def test_evaluate_and_route_llm_exception():
    print("--- Target 3: evaluate_and_route LLM Exception String Guard ---")
    from verifier import evaluate_and_route

    llm_fail_str = (
        "LLM Generation Exception: All priority models failed. "
        "Last error: [gemini-flash] Connection timeout"
    )

    # Must not raise; must return QUARANTINE without a split/parse error.
    try:
        result = evaluate_and_route(llm_fail_str)
    except Exception as e:
        raise AssertionError(
            f"FAIL: evaluate_and_route raised an exception on LLM error string: {e}"
        )

    assert result["status"] == "QUARANTINE", \
        f"FAIL: Expected QUARANTINE, got {result['status']}"
    assert result["reason"] == llm_fail_str, \
        f"FAIL: reason mismatch: {result['reason']}"
    print("  [PASS] LLM exception string safely quarantined without parse error")
    print("  => All Target 3 checks PASSED\n")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    test_are_results_equal()
    test_ast_canonicalizer_closure_scope()
    test_evaluate_and_route_llm_exception()
    print("ALL PATCH TARGET UNIT TESTS PASSED SUCCESSFULLY!")
    sys.exit(0)
