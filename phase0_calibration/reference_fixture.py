"""reference_fixture.py

Deterministic Pre-Experiment Calibration Fixture for Reference Structural-Independence Gate.

Evaluates normalized AST function-only similarity across 6 representative classes of code pairs:
- Class 1: exact duplicates (Expected: 1.0, reject)
- Class 2: identifier/literal-renamed duplicates (Expected: >= 0.85, reject)
- Class 3: superficial formatting/comment variants (Expected: >= 0.85, reject)
- Class 4: same computational strategy, independently written (Expected: high similarity, reject/near-isomorphic)
- Class 5: structurally distinct correct algorithms (Expected: < 0.85, pass)
- Class 6: structurally close but genuinely different algorithms (Expected: < 0.85, pass)

This fixture is an engineering calibration artifact independent of training, calibration (MBPP),
and evaluation (HumanEval/MBPP-test) datasets.
"""

import hashlib
import os
import sys
from typing import Any, Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from quality_gates import check_reference_independence


CALIBRATION_PAIRS: List[Dict[str, Any]] = [
    # Class 1: Exact Duplicates
    {
        "pair_id": "PAIR_01_EXACT_DUP_1",
        "category": "Class 1: Exact duplicate",
        "expected_verdict": "REJECT",
        "solve_opt": (
            "def solve(nums, k):\n"
            "    res = []\n"
            "    for x in nums:\n"
            "        if x > 0:\n"
            "            res.append(x)\n"
            "    return res\n"
        ),
        "solve_brute": (
            "def solve_brute(nums, k):\n"
            "    res = []\n"
            "    for x in nums:\n"
            "        if x > 0:\n"
            "            res.append(x)\n"
            "    return res\n"
        ),
    },
    {
        "pair_id": "PAIR_02_EXACT_DUP_2",
        "category": "Class 1: Exact duplicate",
        "expected_verdict": "REJECT",
        "solve_opt": (
            "from collections import deque\n"
            "def solve(arr):\n"
            "    q = deque(arr)\n"
            "    total = 0\n"
            "    while q:\n"
            "        total += q.popleft()\n"
            "    return total\n"
        ),
        "solve_brute": (
            "from collections import deque\n"
            "def solve_brute(arr):\n"
            "    q = deque(arr)\n"
            "    total = 0\n"
            "    while q:\n"
            "        total += q.popleft()\n"
            "    return total\n"
        ),
    },

    # Class 2: Identifier/literal-renamed duplicates
    {
        "pair_id": "PAIR_03_RENAMED_DUP_1",
        "category": "Class 2: Identifier/literal-renamed duplicate",
        "expected_verdict": "REJECT",
        "solve_opt": (
            "def solve(arr, limit):\n"
            "    output = []\n"
            "    for element in arr:\n"
            "        if element > limit:\n"
            "            output.append(element)\n"
            "    return output\n"
        ),
        "solve_brute": (
            "def solve_brute(items, threshold):\n"
            "    res = []\n"
            "    for val in items:\n"
            "        if val > threshold:\n"
            "            res.append(val)\n"
            "    return res\n"
        ),
    },
    {
        "pair_id": "PAIR_04_RENAMED_DUP_2",
        "category": "Class 2: Identifier/literal-renamed duplicate",
        "expected_verdict": "REJECT",
        "solve_opt": (
            "def solve(matrix, target):\n"
            "    count = 0\n"
            "    for row in matrix:\n"
            "        for cell in row:\n"
            "            if cell == target:\n"
            "                count += 1\n"
            "    return count\n"
        ),
        "solve_brute": (
            "def solve_brute(grid, key):\n"
            "    ans = 0\n"
            "    for r in grid:\n"
            "        for c in r:\n"
            "            if c == key:\n"
            "                ans += 1\n"
            "    return ans\n"
        ),
    },

    # Class 3: Superficial formatting / comment variants
    {
        "pair_id": "PAIR_05_FORMATTING_COMMENT_VARIANT",
        "category": "Class 3: Superficial formatting/comment variant",
        "expected_verdict": "REJECT",
        "solve_opt": (
            "def solve(nums, k):\n"
            "    # Filter positive elements meeting condition\n"
            "    res = []\n"
            "    for x in nums:\n"
            "        # Check positive\n"
            "        if x > 0:\n"
            "            res.append(x)\n"
            "    return res\n"
        ),
        "solve_brute": (
            "def solve_brute(nums, k):\n"
            "\n"
            "    res = []\n"
            "\n"
            "    for x in nums:\n"
            "\n"
            "        if x > 0:\n"
            "            res.append(x)\n"
            "\n"
            "    return res\n"
        ),
    },
    {
        "pair_id": "PAIR_06_DOCSTRING_VARIANT",
        "category": "Class 3: Superficial formatting/comment variant",
        "expected_verdict": "REJECT",
        "solve_opt": (
            "def solve(nums, k):\n"
            "    \"\"\"Optimal solution docstring with parameter notes.\"\"\"\n"
            "    res = []\n"
            "    for x in nums:\n"
            "        if x > 0:\n"
            "            res.append(x)\n"
            "    return res\n"
        ),
        "solve_brute": (
            "def solve_brute(nums, k):\n"
            "    \"\"\"Brute-force reference docstring with alternative notes.\"\"\"\n"
            "    res = []\n"
            "    for x in nums:\n"
            "        if x > 0:\n"
            "            res.append(x)\n"
            "    return res\n"
        ),
    },

    # Class 4: Same computational strategy, independently written
    {
        "pair_id": "PAIR_07_SAME_STRATEGY_DEQUE",
        "category": "Class 4: Same computational strategy independently written",
        "expected_verdict": "REJECT",
        "solve_opt": (
            "from collections import deque\n"
            "def solve(nums, k):\n"
            "    q = deque()\n"
            "    res = []\n"
            "    for i, x in enumerate(nums):\n"
            "        if q and q[0] <= i - k:\n"
            "            q.popleft()\n"
            "        while q and nums[q[-1]] <= x:\n"
            "            q.pop()\n"
            "        q.append(i)\n"
            "        if i >= k - 1:\n"
            "            res.append(nums[q[0]])\n"
            "    return res\n"
        ),
        "solve_brute": (
            "from collections import deque\n"
            "def solve_brute(nums, k):\n"
            "    d = deque()\n"
            "    ans = []\n"
            "    for idx, val in enumerate(nums):\n"
            "        if d and d[0] <= idx - k:\n"
            "            d.popleft()\n"
            "        while d and nums[d[-1]] <= val:\n"
            "            d.pop()\n"
            "        d.append(idx)\n"
            "        if idx >= k - 1:\n"
            "            ans.append(nums[d[0]])\n"
            "    return ans\n"
        ),
    },

    # Class 5: Structurally distinct correct algorithms
    {
        "pair_id": "PAIR_08_DISTINCT_DEQUE_VS_NAIVE",
        "category": "Class 5: Structurally distinct algorithms",
        "expected_verdict": "PASS",
        "solve_opt": (
            "from collections import deque\n"
            "def solve(nums, k):\n"
            "    q = deque()\n"
            "    res = []\n"
            "    for i, x in enumerate(nums):\n"
            "        if q and q[0] <= i - k:\n"
            "            q.popleft()\n"
            "        while q and nums[q[-1]] <= x:\n"
            "            q.pop()\n"
            "        q.append(i)\n"
            "        if i >= k - 1:\n"
            "            res.append(nums[q[0]])\n"
            "    return res\n"
        ),
        "solve_brute": (
            "def solve_brute(nums, k):\n"
            "    out = []\n"
            "    for i in range(len(nums) - k + 1):\n"
            "        cur_max = nums[i]\n"
            "        for j in range(i, i + k):\n"
            "            if nums[j] > cur_max:\n"
            "                cur_max = nums[j]\n"
            "        out.append(cur_max)\n"
            "    return out\n"
        ),
    },
    {
        "pair_id": "PAIR_09_DISTINCT_DP_VS_RECURSION",
        "category": "Class 5: Structurally distinct algorithms",
        "expected_verdict": "PASS",
        "solve_opt": (
            "def solve(nums, k):\n"
            "    prefix = [0]\n"
            "    for x in nums:\n"
            "        prefix.append(prefix[-1] + x)\n"
            "    ans = 0\n"
            "    for i in range(len(nums)):\n"
            "        for j in range(i, len(nums)):\n"
            "            if prefix[j + 1] - prefix[i] <= k:\n"
            "                ans = max(ans, j - i + 1)\n"
            "    return ans\n"
        ),
        "solve_brute": (
            "def solve_brute(nums, k):\n"
            "    def search(idx, current_sum, length):\n"
            "        if current_sum > k:\n"
            "            return 0\n"
            "        if idx == len(nums):\n"
            "            return length\n"
            "        take = search(idx + 1, current_sum + nums[idx], length + 1)\n"
            "        skip = search(idx + 1, 0, 0)\n"
            "        return max(take, skip)\n"
            "    return search(0, 0, 0)\n"
        ),
    },
    {
        "pair_id": "PAIR_10_DISTINCT_MONOTONIC_STACK_VS_PAIRWISE",
        "category": "Class 5: Structurally distinct algorithms",
        "expected_verdict": "PASS",
        "solve_opt": (
            "def solve(heights):\n"
            "    stack = []\n"
            "    max_area = 0\n"
            "    for i, h in enumerate(heights):\n"
            "        while stack and heights[stack[-1]] > h:\n"
            "            top = stack.pop()\n"
            "            w = i if not stack else i - stack[-1] - 1\n"
            "            max_area = max(max_area, heights[top] * w)\n"
            "        stack.append(i)\n"
            "    return max_area\n"
        ),
        "solve_brute": (
            "def solve_brute(heights):\n"
            "    max_area = 0\n"
            "    n = len(heights)\n"
            "    for i in range(n):\n"
            "        for j in range(i, n):\n"
            "            min_h = min(heights[i : j + 1])\n"
            "            max_area = max(max_area, min_h * (j - i + 1))\n"
            "    return max_area\n"
        ),
    },

    # Class 6: Structurally close but genuinely different algorithms
    {
        "pair_id": "PAIR_11_CLOSE_BFS_VS_DFS",
        "category": "Class 6: Structurally close but genuinely different algorithms",
        "expected_verdict": "PASS",
        "solve_opt": (
            "from collections import deque\n"
            "def solve(graph, start, target):\n"
            "    q = deque([(start, 0)])\n"
            "    visited = {start}\n"
            "    while q:\n"
            "        node, d = q.popleft()\n"
            "        if node == target:\n"
            "            return d\n"
            "        for neighbor in graph.get(node, []):\n"
            "            if neighbor not in visited:\n"
            "                visited.add(neighbor)\n"
            "                q.append((neighbor, d + 1))\n"
            "    return -1\n"
        ),
        "solve_brute": (
            "def solve_brute(graph, start, target):\n"
            "    visited = set()\n"
            "    def dfs(u, depth):\n"
            "        if u == target:\n"
            "            return depth\n"
            "        visited.add(u)\n"
            "        best = 999999\n"
            "        for v in graph.get(u, []):\n"
            "            if v not in visited:\n"
            "                best = min(best, dfs(v, depth + 1))\n"
            "        visited.remove(u)\n"
            "        return best\n"
            "    ans = dfs(start, 0)\n"
            "    return ans if ans != 999999 else -1\n"
        ),
    },
]


def run_calibration_fixture(threshold: float = 0.85) -> Dict[str, Any]:
    """Runs the deterministic calibration fixture against the reference-independence gate."""
    results = []
    category_scores: Dict[str, List[float]] = {}
    pass_count = 0
    fail_count = 0

    for pair in CALIBRATION_PAIRS:
        gate_res = check_reference_independence(
            solve_opt=pair["solve_opt"],
            solve_brute=pair["solve_brute"],
            threshold=threshold,
        )
        score = gate_res["similarity_score"]
        actual_verdict = gate_res["status"]
        matches_expected = actual_verdict == pair["expected_verdict"]

        if matches_expected:
            pass_count += 1
        else:
            fail_count += 1

        cat = pair["category"]
        if cat not in category_scores:
            category_scores[cat] = []
        category_scores[cat].append(score)

        results.append({
            "pair_id": pair["pair_id"],
            "category": cat,
            "expected_verdict": pair["expected_verdict"],
            "actual_verdict": actual_verdict,
            "similarity_score": score,
            "threshold": threshold,
            "matches_expected": matches_expected,
            "reason": gate_res["reason"],
        })

    # Compute deterministic fixture hash
    fixture_content = "".join(f"{p['pair_id']}:{p['solve_opt']}:{p['solve_brute']}" for p in CALIBRATION_PAIRS)
    fixture_hash = hashlib.sha256(fixture_content.encode("utf-8")).hexdigest()[:16]

    return {
        "fixture_hash": fixture_hash,
        "total_pairs": len(CALIBRATION_PAIRS),
        "passed_assertions": pass_count,
        "failed_assertions": fail_count,
        "all_passed": fail_count == 0,
        "threshold": threshold,
        "results": results,
        "category_summary": {
            cat: {
                "min": min(scores),
                "max": max(scores),
                "mean": round(sum(scores) / len(scores), 4),
                "scores": scores,
            }
            for cat, scores in category_scores.items()
        },
    }


if __name__ == "__main__":
    report = run_calibration_fixture()
    print("=== REFERENCE INDEPENDENCE THRESHOLD CALIBRATION REPORT ===")
    print(f"Fixture Hash: {report['fixture_hash']}")
    print(f"Threshold: {report['threshold']}")
    print(f"All Assertions Passed: {report['all_passed']} ({report['passed_assertions']}/{report['total_pairs']})\n")
    print("--- Detailed Results ---")
    for r in report["results"]:
        mark = "OK" if r["matches_expected"] else "MISMATCH"
        print(f"[{mark:<8}] {r['pair_id']:<35} | Expected: {r['expected_verdict']:<6} | Actual: {r['actual_verdict']:<6} | Score: {r['similarity_score']:.4f} | {r['category']}")
    print("\n--- Score Distribution by Category ---")
    for cat, stats in report["category_summary"].items():
        print(f"{cat:<55}: min={stats['min']:.4f}, max={stats['max']:.4f}, mean={stats['mean']:.4f}")

