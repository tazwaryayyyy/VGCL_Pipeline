import math
import collections
import heapq
import bisect
import itertools
import random
import time

def solve(arr: list, k: int) -> int:
    """
    Optimal production code.
    Target Complexity: O(N) time and O(N) space using Monotonic Queue DP.
    Must pass within 2.0s for N = 10^5.
    """
    n = len(arr)
    if n == 0:
        return 0
    if n == 1:
        return arr[0]

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
    """
    Canonical naive implementation.
    Target Complexity: O(N * k).
    """
    n = len(arr)
    if n == 0:
        return 0
    if n == 1:
        return arr[0]

    dp = [float('-inf')] * n
    dp[0] = arr[0]

    for i in range(n):
        if dp[i] == float('-inf'):
            continue
        max_reach = min(n, i + k + 1)
        for j in range(i + 1, max_reach):
            if dp[i] + arr[j] > dp[j]:
                dp[j] = dp[i] + arr[j]

    return dp[-1]


def generate_test_cases(seed: int, scale: str) -> list:
    """
    Deterministic test case generator.
    - seed: int for random.seed()
    - scale: "small" (N in [1, 15]) or "large" (N in [10^4, 10^5])

    Returns a list of tuples (arr, k).
    """
    random.seed(seed)
    test_cases = []

    for _ in range(10):
        if scale == "small":
            n = random.randint(1, 15)
            k = random.randint(1, n)
            arr = [random.randint(-100, 100) for _ in range(n)]
        else:
            n = random.randint(10000, 100000)
            k = random.randint(1, n)
            arr = [random.randint(-10**6, 10**6) for _ in range(n)]
        test_cases.append((arr, k))

    return test_cases


def verify_implementations():
    print("--- Verifying Small Scale Test Cases ---")
    small_cases = generate_test_cases(seed=42, scale="small")
    for idx, args in enumerate(small_cases, 1):
        opt_res = solve(*args)
        brute_res = solve_brute(*args)
        assert opt_res == brute_res, f"Mismatch on small case {idx}: opt={opt_res}, brute={brute_res}"
        print(f"Small Case {idx:2d}: Passed | Result: {opt_res}")

    print("\n--- Verifying Large Scale Performance ---")
    large_cases = generate_test_cases(seed=42, scale="large")
    for idx, args in enumerate(large_cases, 1):
        arr, k = args
        start_time = time.perf_counter()
        opt_res = solve(arr, k)
        elapsed = time.perf_counter() - start_time
        assert elapsed < 2.0, f"Performance bottleneck on large case {idx}: {elapsed:.4f}s"
        print(f"Large Case {idx:2d} (N={len(arr):6d}, k={k:6d}): Passed in {elapsed:.4f}s | Result: {opt_res}")

    print("\nALL VERIFICATION TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    verify_implementations()
