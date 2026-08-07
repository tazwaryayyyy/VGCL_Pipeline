import sys
import os
import math
from verifier import run_full_verification, verify_ast_anti_cheat, _are_results_equal
from dedup import CorpusBalanceEngine, compute_jaccard_similarity, get_canonical_tokens, get_ngrams
from generator import parse_delimiters

def test_ast_security_sandbox():
    print("--- 1. Testing AST Security Visitor (Sandbox Hardening) ---")
    malicious_code_1 = """
import os
def solve(*args):
    os.system("echo HACKED")
    return 0
def solve_brute(*args): return 0
def generate_test_cases(seed, scale): return [(1,)]
"""
    ok, msg = verify_ast_anti_cheat(malicious_code_1)
    assert not ok and "Forbidden module import" in msg, f"Security check failed to block os import: {msg}"
    print("Malicious import 'os' blocked: PASSED")

    malicious_code_2 = """
def solve(*args):
    f = open("/etc/passwd", "r")
    return 0
def solve_brute(*args): return 0
def generate_test_cases(seed, scale): return [(1,)]
"""
    ok, msg = verify_ast_anti_cheat(malicious_code_2)
    assert not ok and "Forbidden builtin access" in msg, f"Security check failed to block open: {msg}"
    print("Forbidden builtin 'open' blocked: PASSED")

def test_ast_ngram_padding_and_coverage():
    print("\n--- 2. Testing AST Canonicalization & Uniform N-Gram Padding ---")
    # Small function (tokens < 4)
    tokens_small_1 = ["v_0", "v_1"]
    tokens_small_2 = ["v_0", "v_1", "v_2"]
    sim_small = compute_jaccard_similarity(tokens_small_1, tokens_small_2, k=4)
    print(f"Similarity for small token sequences with padding: {sim_small:.4f}")
    assert sim_small > 0.0, "Small token padding should prevent false 0.0 Jaccard similarity!"

    # Async function & Class AST coverage
    code_async_1 = "async def solve(x): return x"
    code_async_2 = "async def solve(y): return y"
    tok_1 = get_canonical_tokens(code_async_1)
    tok_2 = get_canonical_tokens(code_async_2)
    assert tok_1 == tok_2, "AsyncFunctionDef canonicalization failed to normalize parameter names!"
    print("Async & Class AST Canonicalization: PASSED")

def test_differential_equality():
    print("\n--- 3. Testing Differential Floating-Point & NaN Equality ---")
    assert _are_results_equal(1.0000000000000002, 1.0), "Float precision equality check failed"
    assert _are_results_equal(float('nan'), float('nan')), "NaN equality check failed"
    assert _are_results_equal([1.0, float('nan')], [1.0000000000000001, float('nan')]), "Nested list NaN check failed"
    assert not _are_results_equal(1.5, 2.5), "Different floats returned True"
    print("Differential Equality Check: PASSED")

def test_delimiter_parser():
    print("\n--- 4. Testing Delimiter Parser Header Indexing & Fence Stripping ---")
    raw_llm_output = """### PROBLEM_SPEC
Target Paradigm: DP
Problem Name: Test Problem
Note: Use ### SOLVE_BRUTE in comments safely.

### SOLVE_OPTIMIZED
```python
def solve(x):
    # Comment mentioning ### TEST_GENERATOR
    return x
```

### SOLVE_BRUTE
```python
def solve_brute(x):
    return x
```

### TEST_GENERATOR
def generate_test_cases(seed, scale): return [(1,)]
"""
    parsed = parse_delimiters(raw_llm_output)
    assert parsed["valid"], f"Parser error: {parsed.get('error')}"
    assert "```python" not in parsed["solve_opt"], "Failed to strip markdown code fences!"
    assert "def solve(x):" in parsed["solve_opt"], "Code content corrupted by parser!"
    print("Delimiter Parser: PASSED")

def test_full_sandbox_verification():
    print("\n--- 5. Testing Full Verification Engine & Non-Blocking IPC ---")
    valid_code = """
import collections

def solve(arr: list, k: int) -> int:
    n = len(arr)
    if n == 0: return 0
    if n == 1: return arr[0]
    dp = [0] * n
    dp[0] = arr[0]
    dq = collections.deque([0])
    for j in range(1, n):
        while dq and dq[0] < j - k:
            dq.popleft()
        dp[j] = arr[j] + dp[dq[0]]
        while dq and dp[dq[-1]] <= dp[j]:
            dq.pop()
        dq.append(j)
    return dp[-1]

def solve_brute(arr: list, k: int) -> int:
    n = len(arr)
    if n == 0: return 0
    if n == 1: return arr[0]
    dp = [float('-inf')] * n
    dp[0] = arr[0]
    for i in range(n):
        if dp[i] == float('-inf'): continue
        max_reach = min(n, i + k + 1)
        for j in range(i + 1, max_reach):
            if dp[i] + arr[j] > dp[j]:
                dp[j] = dp[i] + arr[j]
    return dp[-1]

import random

def generate_test_cases(seed: int, scale: str) -> list:
    random.seed(seed)
    test_cases = []
    for _ in range(5):
        if scale == "small":
            n = random.randint(1, 10)
            k = random.randint(1, n)
            arr = [random.randint(-50, 50) for _ in range(n)]
        else:
            n = random.randint(1000, 5000)
            k = random.randint(1, n)
            arr = [random.randint(-1000, 1000) for _ in range(n)]
        test_cases.append((arr, k))
    return test_cases
"""
    verified, ver_msg, failing_input = run_full_verification(valid_code, seed=42)
    assert verified, f"Verification failed: {ver_msg}"
    print(f"Full Verification Engine Result: PASSED ({ver_msg})")

if __name__ == "__main__":
    test_ast_security_sandbox()
    test_ast_ngram_padding_and_coverage()
    test_differential_equality()
    test_delimiter_parser()
    test_full_sandbox_verification()
    print("\nALL PRODUCTION FIXES & SECURITY AUDIT TEST CASES PASSED SUCCESSFULLY!")
