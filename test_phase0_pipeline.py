"""test_phase0_pipeline.py

Comprehensive Phase 0 Integration Test Suite.
Validates end-to-end functionality across all 8 core Phase 0 modules:
1. failure_mapper.py
2. oracle_refactor.py
3. cfg_dedup.py
4. experiment_ledger.py
5. targeted_generation.py
6. uniform_generation.py
7. transfer_matrix.py
8. statistical_analysis.py
"""

import os
import shutil
import tempfile
import unittest
import numpy as np

from cfg_dedup import StructuralDeduplicationEngine, compute_ast_hashes
from experiment_ledger import ExperimentLedger, LedgerEntry
from failure_mapper import FailureCategory, FailureMapper, classify_failure
from oracle_refactor import VerificationOracle
from statistical_analysis import VGCLStatisticalAnalyzer
from transfer_matrix import TransferMatrixCalculator


class TestPhase0Pipeline(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.test_dir, "test_ledger.db")
        self.jsonl_path = os.path.join(self.test_dir, "test_ledger.jsonl")
        self.quarantine_dir = os.path.join(self.test_dir, "quarantine")

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_failure_mapper(self):
        mapper = FailureMapper(model_version="Qwen2.5-1.5B-Test")
        rec = mapper.record_failure(
            problem_id="TEST_P1",
            prompt="Test prompt",
            generated_code="def solve(): pass",
            execution_result="IndexError: list index out of range",
            exception_type="IndexError",
            is_small_scale=True,
        )
        self.assertEqual(rec.failure_category, FailureCategory.BOUNDS_VIOLATION)
        self.assertEqual(rec.cluster_id, "FC_04_BOUNDS_VIOLATION")

    def test_verification_oracle_success(self):
        oracle = VerificationOracle(quarantine_dir=self.quarantine_dir)
        valid_code = (
            "def solve(nums):\n    return sum(nums)\n\n"
            "def solve_brute(nums):\n    s = 0\n    for x in nums:\n        s += x\n    return s\n\n"
            "import random\n"
            "def generate_test_cases(seed, scale):\n"
            "    random.seed(seed)\n"
            "    if scale == 'small':\n"
            "        return [([random.randint(1, 10) for _ in range(5)],) for _ in range(3)]\n"
            "    else:\n"
            "        return [([random.randint(1, 10) for _ in range(100)],) for _ in range(2)]\n"
        )
        res = oracle.verify(valid_code, seed=42)
        self.assertTrue(res["verified"])
        self.assertEqual(res["confidence"], 1.0)
        self.assertTrue(res["phase_a_pass"])
        self.assertTrue(res["phase_b_pass"])

    def test_verification_oracle_mismatch_quarantine(self):
        oracle = VerificationOracle(quarantine_dir=self.quarantine_dir)
        mismatch_code = (
            "def solve(nums):\n    return sum(nums) + 1\n\n"  # Wrong answer!
            "def solve_brute(nums):\n    return sum(nums)\n\n"
            "import random\n"
            "def generate_test_cases(seed, scale):\n"
            "    random.seed(seed)\n"
            "    if scale == 'small':\n"
            "        return [([random.randint(1, 10) for _ in range(5)],) for _ in range(3)]\n"
            "    else:\n"
            "        return [([random.randint(1, 10) for _ in range(100)],) for _ in range(2)]\n"
        )
        res = oracle.verify(mismatch_code, seed=42)
        self.assertFalse(res["verified"])
        self.assertIsNotNone(res["quarantine_id"])
        self.assertTrue(os.path.exists(os.path.join(self.quarantine_dir, f"{res['quarantine_id']}.json")))

    def test_cfg_deduplication(self):
        engine = StructuralDeduplicationEngine(similarity_threshold=0.70)
        code1 = "def solve(a, b):\n    if a > b:\n        return a\n    return b"
        code2 = "def solve(x, y):\n    if x > y:\n        return x\n    return y"
        code3 = "def solve(arr):\n    return sorted(arr)"

        res1 = engine.register_candidate("P1", code1)
        self.assertFalse(res1.is_duplicate)

        res2 = engine.check_candidate("P2", code2)
        self.assertTrue(res2.is_duplicate)  # Structurally identical!

        res3 = engine.check_candidate("P3", code3)
        self.assertFalse(res3.is_duplicate)  # Structurally distinct!

    def test_experiment_ledger(self):
        ledger = ExperimentLedger(db_path=self.db_path, jsonl_path=self.jsonl_path)
        entry = LedgerEntry(
            problem_id="P_TEST_LEDGER",
            seed=42,
            target_failure_cluster="FC_01_MICRO_CORRECTNESS",
            monotonic_queue_subtype="Type A: sliding-window extrema",
            code_content="def solve(): pass",
        )
        ledger.record_entry(entry)

        entries = ledger.get_entries("FC_01_MICRO_CORRECTNESS")
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["problem_id"], "P_TEST_LEDGER")
        
        balance = ledger.get_cluster_balance_stats()
        self.assertIn("FC_01_MICRO_CORRECTNESS", balance)
        self.assertEqual(balance["FC_01_MICRO_CORRECTNESS"]["sample_count"], 1)

    def test_evaluation_leakage_guard(self):
        mapper = FailureMapper(model_version="Qwen2.5-1.5B-Test")
        mapper.record_failure(
            problem_id="TRAIN_PROB_001",
            prompt="Prompt",
            generated_code="def solve(): pass",
            execution_result="Error",
        )
        # Disjoint set passes
        self.assertTrue(mapper.verify_no_evaluation_leakage(["EVAL_PROB_001", "EVAL_PROB_002"]))
        # Overlapping set raises ValueError
        with self.assertRaises(ValueError):
            mapper.verify_no_evaluation_leakage(["TRAIN_PROB_001", "EVAL_PROB_002"])

    def test_transfer_matrix(self):
        calc = TransferMatrixCalculator()
        clusters = calc.clusters
        base_rates = {c: 0.10 for c in clusters}
        ft_rates = {
            c_k: {e_j: (0.50 if c_k == e_j else 0.15) for e_j in clusters}
            for c_k in clusters
        }
        res = calc.calculate_transfer_matrix(base_rates, ft_rates)
        self.assertGreater(np.mean(list(res.within_cluster.values())), res.cross_cluster_avg)

    def test_statistical_analyzer(self):
        analyzer = VGCLStatisticalAnalyzer(alpha=0.05)
        tgt_runs = [0.45, 0.48, 0.46]
        uni_runs = [0.15, 0.18, 0.16]
        res = analyzer.evaluate_endpoint("Test Endpoint", tgt_runs, uni_runs)
        self.assertTrue(res.statistically_significant)
        self.assertGreater(res.effect_size_cohens_d, 2.0)


if __name__ == "__main__":
    unittest.main()
