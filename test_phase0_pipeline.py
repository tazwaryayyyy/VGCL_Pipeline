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
from pipeline_metrics import PipelineMetrics
from statistical_analysis import VGCLStatisticalAnalyzer
from syntax_repair import SyntaxDiagnostic, check_and_repair
from targeted_generation import SUBTYPES, SubtypeScheduler
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

    def test_calibration_dataset_leakage_guard(self):
        import json
        raw_calib_path = "data/raw_calibration.json"
        if os.path.exists(raw_calib_path):
            with open(raw_calib_path, "r", encoding="utf-8") as f:
                calib_data = json.load(f)
            calib_ids = {item["task_id"] for item in calib_data}
            # Enforce zero HumanEval IDs or MBPP test IDs in calibration set
            humaneval_in_calib = [tid for tid in calib_ids if "HumanEval" in tid or tid.startswith("HumanEval/")]
            mbpp_test_in_calib = [tid for tid in calib_ids if tid.startswith("MBPP/")]
            self.assertEqual(len(humaneval_in_calib), 0, f"HumanEval problems found in calibration set: {humaneval_in_calib}")
            self.assertEqual(len(mbpp_test_in_calib), 0, f"MBPP test problems found in calibration set: {mbpp_test_in_calib}")

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

    # ------------------------------------------------------------------
    # New tests for throughput optimization modules
    # ------------------------------------------------------------------

    def test_syntax_repair_strips_fences(self):
        """Verifies that markdown fence stripping produces valid-parseable code."""
        fenced = (
            "```python\n"
            "def solve(arr):\n"
            "    return sorted(arr)\n"
            "\n"
            "def solve_brute(arr):\n"
            "    res = list(arr)\n"
            "    for i in range(len(res)):\n"
            "        for j in range(i+1, len(res)):\n"
            "            if res[i] > res[j]:\n"
            "                res[i], res[j] = res[j], res[i]\n"
            "    return res\n"
            "\n"
            "import random\n"
            "def generate_test_cases(seed, scale):\n"
            "    random.seed(seed)\n"
            "    if scale == 'small':\n"
            "        return [([1, 2, 3],)]\n"
            "    else:\n"
            "        return [([1] * 100,)]\n"
            "```"
        )
        diag, repaired_code = check_and_repair(fenced)
        self.assertTrue(diag.passed, f"Expected parsed OK after fence strip, got: {diag.error_msg}")
        self.assertTrue(diag.repaired, "Expected repaired=True when fences were stripped")
        self.assertIn("stripped_markdown_fences", diag.repair_applied)
        self.assertEqual(diag.error_category, "none")

    def test_syntax_repair_no_op_on_clean_code(self):
        """Verifies that already-valid code returns repaired=False and passes immediately."""
        clean = (
            "def solve(x):\n"
            "    return x * 2\n"
            "\n"
            "def solve_brute(x):\n"
            "    result = 0\n"
            "    for _ in range(x):\n"
            "        result += 2\n"
            "    return result\n"
            "\n"
            "import random\n"
            "def generate_test_cases(seed, scale):\n"
            "    random.seed(seed)\n"
            "    if scale == 'small':\n"
            "        return [(random.randint(1, 5),) for _ in range(3)]\n"
            "    else:\n"
            "        return [(random.randint(100, 500),) for _ in range(2)]\n"
        )
        diag, returned_code = check_and_repair(clean)
        self.assertTrue(diag.passed)
        self.assertFalse(diag.repaired)
        self.assertEqual(diag.repair_applied, "none")
        self.assertEqual(diag.error_category, "none")
        self.assertEqual(returned_code, clean)

    def test_pipeline_metrics_records_approval(self):
        """Unit test: PipelineMetrics counters are mutated correctly across attempt/approval."""
        m = PipelineMetrics()
        m.record_attempt(subtype="Type A: sliding-window extrema", cluster="FC_01_MICRO_CORRECTNESS")
        m.record_gen_latency(3.5)
        m.record_api_calls(2)
        m.record_syntax_repair()
        m.record_oracle_attempt()
        m.record_phase_a_pass()
        m.record_phase_b_pass()
        m.record_approved(subtype="Type A: sliding-window extrema", cluster="FC_01_MICRO_CORRECTNESS")

        snap = m.snapshot()
        self.assertEqual(snap["attempts"], 1)
        self.assertEqual(snap["approved"], 1)
        self.assertEqual(snap["syntax_repairs"], 1)
        self.assertEqual(snap["oracle_attempts"], 1)
        self.assertEqual(snap["phase_a_passes"], 1)
        self.assertEqual(snap["phase_b_passes"], 1)
        self.assertEqual(snap["approval_rate"], 1.0)
        self.assertEqual(snap["api_calls"], 2)
        self.assertIn("Type A: sliding-window extrema", snap["approval_rate_by_subtype"])
        self.assertEqual(snap["approval_rate_by_subtype"]["Type A: sliding-window extrema"], 1.0)
        self.assertIn("FC_01_MICRO_CORRECTNESS", snap["approval_rate_by_cluster"])

    def test_pipeline_metrics_denominators(self):
        """Known event sequence yields exact rate denominators."""
        m = PipelineMetrics()

        # Attempt 1: generation/block parsing failure, so syntax was never attempted.
        m.record_attempt(subtype="A", cluster="FC")
        m.record_api_calls(1)
        m.record_gen_error()

        # Attempt 2: generation succeeded, syntax failed before oracle.
        m.record_attempt(subtype="A", cluster="FC")
        m.record_api_calls(2)
        m.record_syntax_failure()

        # Attempt 3: verified by oracle, then rejected by deduplication.
        m.record_attempt(subtype="B", cluster="FC")
        m.record_api_calls(2)
        m.record_oracle_attempt()
        m.record_phase_a_pass()
        m.record_phase_b_pass()
        m.record_dedup_rejection()

        # Attempt 4: accepted into the ledger.
        m.record_attempt(subtype="B", cluster="FC")
        m.record_api_calls(2)
        m.record_oracle_attempt()
        m.record_phase_a_pass()
        m.record_phase_b_pass()
        m.record_approved(subtype="B", cluster="FC")

        snap = m.snapshot()
        self.assertEqual(snap["attempts"], 4)
        self.assertEqual(snap["syntax_attempts"], 3)
        self.assertEqual(snap["syntax_passes"], 2)
        self.assertEqual(snap["dedup_submissions"], 2)
        self.assertEqual(snap["approval_rate"], 0.25)
        self.assertAlmostEqual(snap["syntax_pass_rate"], 2 / 3, places=6)
        self.assertEqual(snap["phase_a_pass_rate"], 1.0)
        self.assertEqual(snap["phase_b_pass_rate"], 1.0)
        self.assertEqual(snap["dedup_rejection_rate"], 0.5)
        self.assertEqual(snap["api_calls_per_approved"], 7.0)

    def test_subtype_scheduler_seed_determinism_and_batch_coverage(self):
        """Same seed path is deterministic and planned batch slots cover subtypes."""
        scheduler_a = SubtypeScheduler(subtypes=SUBTYPES)
        scheduler_b = SubtypeScheduler(subtypes=SUBTYPES)

        planned_a = {}
        planned_b = {}
        seq_a = []
        seq_b = []
        for seed in range(1001, 1001 + len(SUBTYPES)):
            st_a = scheduler_a.select(planned_a, seed)
            st_b = scheduler_b.select(planned_b, seed)
            seq_a.append(st_a)
            seq_b.append(st_b)
            planned_a[st_a] = planned_a.get(st_a, 0) + 1
            planned_b[st_b] = planned_b.get(st_b, 0) + 1

        self.assertEqual(seq_a, seq_b)
        self.assertEqual(seq_a, SUBTYPES)
        self.assertEqual(set(seq_a), set(SUBTYPES))

    def test_oracle_returns_rejection_diagnostics(self):
        """Verifies that the new rejection_diagnostics key is present on both failed and
        successful oracle results, and that its structure is stable and machine-readable."""
        oracle = VerificationOracle(quarantine_dir=self.quarantine_dir)

        # Case 1: mismatching code should produce a non-verified result with diagnostics
        mismatch_code = (
            "def solve(nums):\n    return sum(nums) + 99\n\n"
            "def solve_brute(nums):\n    return sum(nums)\n\n"
            "import random\n"
            "def generate_test_cases(seed, scale):\n"
            "    random.seed(seed)\n"
            "    if scale == 'small':\n"
            "        return [([random.randint(1, 5) for _ in range(3)],) for _ in range(3)]\n"
            "    else:\n"
            "        return [([random.randint(1, 5) for _ in range(100)],) for _ in range(2)]\n"
        )
        res_fail = oracle.verify(mismatch_code, seed=42)
        self.assertFalse(res_fail["verified"])
        self.assertIn("rejection_diagnostics", res_fail)
        diag_fail = res_fail["rejection_diagnostics"]
        # Required keys must all be present
        for key in [
            "syntax_category", "parse_failure_location", "phase_a_mismatch_type",
            "phase_b_failure", "is_timeout", "is_memory", "is_duplicate",
            "is_security", "is_anti_cheat",
        ]:
            self.assertIn(key, diag_fail, f"Missing key '{key}' in rejection_diagnostics")
        # A Phase A mismatch should be classified correctly
        self.assertEqual(diag_fail["phase_a_mismatch_type"], "output_mismatch")
        self.assertFalse(diag_fail["is_security"])
        self.assertFalse(diag_fail["is_anti_cheat"])

        # Case 2: valid code should also carry the key (all-"none" sentinel)
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
        res_ok = oracle.verify(valid_code, seed=42)
        self.assertTrue(res_ok["verified"])
        self.assertIn("rejection_diagnostics", res_ok)
        diag_ok = res_ok["rejection_diagnostics"]
        self.assertEqual(diag_ok["syntax_category"], "none")
        self.assertEqual(diag_ok["phase_a_mismatch_type"], "none")
        self.assertEqual(diag_ok["phase_b_failure"], "none")
        self.assertFalse(diag_ok["is_timeout"])
        self.assertFalse(diag_ok["is_memory"])


if __name__ == "__main__":
    unittest.main()
