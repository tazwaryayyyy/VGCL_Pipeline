from verifier import verify_ast_anti_cheat, run_full_verification

def run_smoke_test():
    print("=== STARTING 2-MINUTE SMOKE TEST PROTOCOL ===")
    
    # Test 1: Verify AST Security Visitor Rejection
    payload = """
import os
def solve(*args): return 0
def solve_brute(*args): return 0
def generate_test_cases(seed, scale): return []
"""
    ok, msg = verify_ast_anti_cheat(payload)
    assert not ok and "Forbidden module import" in msg, f"Security check failed! Message: {msg}"
    print("[TEST 1 PASSED] SecurityVisitor AST Sandbox: PASSED")

    # Test 2: Verify Non-Blocking IPC Under Large Errors (100KB string payload mismatch)
    large_fail_code = """
def solve(*args):
    return "X" * 100000
def solve_brute(*args):
    return "Y" * 100000
def generate_test_cases(seed, scale):
    return [(1,)]
"""
    res = run_full_verification(large_fail_code, seed=42)
    ok, msg = res[0], res[1]
    assert not ok, "Verification should have failed on mismatch."
    print(f"[TEST 2 PASSED] IPC Buffer & Non-Blocking Drain: PASSED ({msg})")

    print("\nALL SMOKE TEST PROTOCOL CHECKS PASSED PERFECTLY!")

if __name__ == "__main__":
    run_smoke_test()
