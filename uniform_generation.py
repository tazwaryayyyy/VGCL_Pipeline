"""uniform_generation.py

Uniform Sampling Control Pipeline for VGCL Phase 0.

Serves as the experimental control condition. Utilizes identical generator model,
decoupled 2-pass verification oracle (Invariant III), structural deduplication engine,
experiment ledger schema, client-side pacing, and difficulty calibration as targeted_generation.py.

The ONLY experimental variable that differs is curriculum targeting: uniform_generation.py
samples uniformly across failure clusters without failure-targeted prompting or stratification.
"""

import argparse
import hashlib
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple
from dotenv import load_dotenv

from cfg_dedup import StructuralDeduplicationEngine
from experiment_ledger import ExperimentLedger, LedgerEntry
from oracle_refactor import VerificationOracle
from pipeline_metrics import PipelineMetrics
from quality_gates import check_reference_independence, validate_subtype
from syntax_repair import (
    check_and_repair,
    identify_malformed_component,
    regenerate_syntax_component,
)
from targeted_generation import _call_llm_paced, calibrate_empirical_difficulty, client_side_pace, parse_block_content

load_dotenv()

UNIFORM_CLUSTERS = [
    "FC_01_MICRO_CORRECTNESS",
    "FC_02_SCALE_DIVERGENCE",
    "FC_03_TERMINATION",
    "FC_04_BOUNDS_VIOLATION",
    "FC_05_RESOURCE_FAILURE",
    "FC_06_FORMAT_CONTRACT",
    "FC_07_STATE_INCONSISTENCY",
    "FC_08_UNKNOWN",
]


def generate_uniform_candidate(
    cluster_id: str,
    seed: int,
) -> Tuple[bool, Dict[str, str], str, float, int]:
    """
    Decoupled 2-Pass Generation Pipeline (Control Condition - Invariant V).
    Pass 1: Generate PROBLEM_SPEC, SOLVE_OPTIMIZED, and TEST_GENERATOR uniformly without failure targeting.
    Pass 2: Isolated call receiving ONLY PROBLEM_SPEC to write naive/brute-force SOLVE_BRUTE.
    """
    system_pass1 = (
        "You are a principal competitive programming problem architect. Design a self-contained, "
        "mathematically sound algorithmic problem package.\n\n"
        "### OUTPUT FORMAT CONTRACT (Pass 1):\n"
        "Output EXACTLY three headers:\n\n"
        "### PROBLEM_SPEC\n"
        "<Problem description, input/output formats, scale constraints>\n\n"
        "### SOLVE_OPTIMIZED\n"
        "def solve(...):\n"
        "    <Optimal implementation>\n\n"
        "### TEST_GENERATOR\n"
        "import random\n"
        "def generate_test_cases(seed: int, scale: str) -> list:\n"
        "    # Line 1 MUST be: random.seed(seed)\n"
        "    # Return list of argument tuples matching 'solve' positional arguments\n"
        "    # scale=='small': N <= 8; scale=='large': N >= 100\n"
    )
    user_pass1 = f"Generate a general algorithmic problem package (Seed {seed})."

    raw_p1, model1, lat1, ret1 = _call_llm_paced(system_pass1, user_pass1)

    spec = parse_block_content(raw_p1, "### PROBLEM_SPEC")
    solve_opt = parse_block_content(raw_p1, "### SOLVE_OPTIMIZED")
    test_gen = parse_block_content(raw_p1, "### TEST_GENERATOR")

    if not spec or not solve_opt or not test_gen:
        return False, {}, model1, lat1, ret1

    # Pass 2: Isolated Reference Implementation (Oracle Independence - Invariant III)
    system_pass2 = (
        "You are an independent algorithmic verification reference generator. "
        "Your goal is to write a simple, naive, brute-force solution (solve_brute) for the provided problem spec.\n\n"
        "### STRICT BRUTE-FORCE CONSTRAINTS:\n"
        "1. MUST use exact same positional function signature and return type as described in the spec.\n"
        "2. MUST use exhaustive search, naive loops, or unoptimized recursion.\n"
        "3. MUST NOT use memoization, dynamic programming, or advanced data structures.\n"
        "4. DO NOT attempt to write optimal code.\n\n"
        "### OUTPUT FORMAT:\n"
        "### SOLVE_BRUTE\n"
        "def solve_brute(...):\n"
        "    ...\n"
    )
    user_pass2 = f"Here is the problem specification:\n\n{spec}\n\nWrite 'solve_brute' for this spec."

    raw_p2, model2, lat2, ret2 = _call_llm_paced(system_pass2, user_pass2)
    solve_brute = parse_block_content(raw_p2, "### SOLVE_BRUTE")

    if not solve_brute:
        return False, {}, model2, lat1 + lat2, ret1 + ret2

    blocks = {
        "spec": spec,
        "solve_opt": solve_opt,
        "solve_brute": solve_brute,
        "test_gen": test_gen,
    }
    return True, blocks, model1, lat1 + lat2, ret1 + ret2


def run_uniform_generation_pipeline(
    num_samples: int = 5,
    db_path: str = "data/uniform_experiment_ledger.db",
    jsonl_path: str = "data/uniform_approved_curriculum.jsonl",
    batch_size: int = 1,
    metrics: Optional[PipelineMetrics] = None,
) -> None:
    """
    Uniform sampling control generation loop.

    Args:
        num_samples: Number of verified, deduplicated samples to accept.
        db_path:     SQLite ledger path for the uniform condition.
        jsonl_path:  JSONL output path for the uniform condition.
        batch_size:  Number of parallel generation calls per outer iteration.
                     Default ``1`` preserves the original serial behaviour.
        metrics:     Optional ``PipelineMetrics`` instance.  When ``None``,
                     a private instance is created and flushed on completion.
    """
    print("=== STARTING UNIFORM SAMPLING CONTROL GENERATION PIPELINE ===")

    oracle = VerificationOracle(quarantine_dir="data/quarantine_uniform")
    dedup = StructuralDeduplicationEngine(similarity_threshold=0.70)
    ledger = ExperimentLedger(db_path=db_path, jsonl_path=jsonl_path)

    if metrics is None:
        metrics = PipelineMetrics()

    seed = 5000
    accepted = 0

    while accepted < num_samples:
        # Pre-compute seeds for all items in this batch before dispatch
        batch_seeds = [seed + i + 1 for i in range(batch_size)]
        seed += batch_size

        # Cluster IDs for this batch (round-robin, unchanged from original)
        batch_clusters = [
            UNIFORM_CLUSTERS[(accepted + i) % len(UNIFORM_CLUSTERS)]
            for i in range(batch_size)
        ]

        def _generate_one_uniform(
            cluster_id: str,
            b_seed: int,
        ) -> Tuple[int, str, bool, Dict[str, str], str, float, int]:
            """Worker thunk executed in the thread pool."""
            ok, blocks, model_name, latency, retries = generate_uniform_candidate(
                cluster_id=cluster_id,
                seed=b_seed,
            )
            return b_seed, cluster_id, ok, blocks, model_name, latency, retries

        effective_batch = min(batch_size, num_samples - accepted)
        futures_map = {}
        with ThreadPoolExecutor(max_workers=effective_batch) as executor:
            for i in range(effective_batch):
                fut = executor.submit(
                    _generate_one_uniform,
                    batch_clusters[i],
                    batch_seeds[i],
                )
                futures_map[fut] = (batch_seeds[i], batch_clusters[i])

            results_by_seed: Dict[int, Any] = {}
            for fut in as_completed(futures_map):
                b_seed, cluster_id = futures_map[fut]
                try:
                    result = fut.result()
                except Exception as exc:
                    result = (b_seed, cluster_id, False, {}, "failed", 0.0, 0)
                    print(f"  [X] Batch worker exception (seed {b_seed}): {exc}")
                results_by_seed[b_seed] = result

        # Process batch results in deterministic seed order
        for b_seed in sorted(results_by_seed.keys()):
            if accepted >= num_samples:
                break

            b_seed_val, cluster_id, ok, blocks, model_name, latency, api_calls = results_by_seed[b_seed]

            metrics.record_attempt(cluster=cluster_id)
            metrics.record_gen_latency(latency)
            metrics.record_initial_generation_api_calls(api_calls)

            print(
                f"\n[+] Generating Uniform Sample [{accepted + 1}/{num_samples}] "
                f"| Category: '{cluster_id}' | Seed: {b_seed_val}"
            )

            if not ok:
                print("  [X] Decoupled Generation Error: Block parsing failed.")
                metrics.record_gen_error()
                continue

            full_code = (
                f"{blocks['solve_opt']}\n\n{blocks['solve_brute']}\n\n{blocks['test_gen']}"
            )
            prompt_hash = hashlib.sha256(blocks["spec"].encode("utf-8")).hexdigest()[:16]

            # 0. Pre-oracle syntax validation + controlled 1-shot component retry
            syn_diag, full_code = check_and_repair(full_code)
            if not syn_diag.passed:
                malformed_comp, comp_diag = identify_malformed_component(blocks)
                if malformed_comp:
                    err_msg = comp_diag.error_msg if comp_diag else syn_diag.error_msg
                    print(f"  [~] Syntax Failure in '{malformed_comp}'. Attempting 1-shot component regeneration...")
                    retry_ok, new_blocks, retry_prov, retry_calls = regenerate_syntax_component(
                        component_name=malformed_comp,
                        blocks=blocks,
                        cluster_id=cluster_id,
                        subtype=cluster_id,
                        seed=b_seed_val,
                        llm_caller=_call_llm_paced,
                        original_error=err_msg,
                    )
                    metrics.record_syntax_retry_api_calls(retry_calls)
                    if retry_ok:
                        blocks = new_blocks
                        full_code = f"{blocks['solve_opt']}\n\n{blocks['solve_brute']}\n\n{blocks['test_gen']}"
                        syn_diag, full_code = check_and_repair(full_code)
                        if syn_diag.passed:
                            metrics.record_syntax_retry_success()
                            print(f"  [✓] Syntax Retry Succeeded for component '{malformed_comp}'")

                if not syn_diag.passed:
                    print(
                        f"  [X] Syntax Validation Failed [{syn_diag.error_category}] "
                        f"(line {syn_diag.error_line}): {syn_diag.error_msg[:120]}"
                    )
                    oracle.quarantine(
                        seed=b_seed_val,
                        prompt_hash=prompt_hash,
                        generator_model=model_name,
                        generator_version="v1.0",
                        optimized_code=blocks.get("solve_opt", ""),
                        reference_code=blocks.get("solve_brute", ""),
                        test_generator_code=blocks.get("test_gen", ""),
                        quarantine_reason=f"Syntax Failure: {syn_diag.error_category} ({syn_diag.error_msg[:100]})",
                    )
                    metrics.record_syntax_failure()
                    continue

            if syn_diag.repaired:
                print(f"  [~] Syntax Repair Applied: {syn_diag.repair_applied}")
                metrics.record_syntax_repair()

            # 1. Reference Structural-Independence Gate
            ref_ind = check_reference_independence(blocks["solve_opt"], blocks["solve_brute"])
            if ref_ind["status"] == "REJECT":
                print(
                    f"  [X] Reference Independence Failed (Similarity {ref_ind['similarity_score']} >= {ref_ind['threshold']}): {ref_ind['reason']}"
                )
                oracle.quarantine(
                    seed=b_seed_val,
                    prompt_hash=prompt_hash,
                    generator_model=model_name,
                    generator_version="v1.0",
                    optimized_code=blocks.get("solve_opt", ""),
                    reference_code=blocks.get("solve_brute", ""),
                    test_generator_code=blocks.get("test_gen", ""),
                    quarantine_reason=f"Reference Independence Violation: {ref_ind['reason']} (Score: {ref_ind['similarity_score']})",
                    execution_traces={"reference_independence": ref_ind},
                )
                metrics.record_reference_independence_rejection()
                continue

            # 2. Subtype Validity Gate (validates if subtype starts with Type A-F; passes if general/NOT_VALIDATABLE)
            st_val = validate_subtype(cluster_id, blocks["spec"], blocks["solve_opt"])
            if st_val["status"] == "REJECT":
                print(f"  [X] Subtype Validation Failed for '{cluster_id}': {st_val['reason']}")
                oracle.quarantine(
                    seed=b_seed_val,
                    prompt_hash=prompt_hash,
                    generator_model=model_name,
                    generator_version="v1.0",
                    optimized_code=blocks.get("solve_opt", ""),
                    reference_code=blocks.get("solve_brute", ""),
                    test_generator_code=blocks.get("test_gen", ""),
                    quarantine_reason=f"Subtype Validation Violation: {st_val['reason']} (Checks: {', '.join(st_val['checks'])})",
                    execution_traces={"subtype_validation": st_val},
                )
                metrics.record_subtype_rejection()
                continue

            # 3. Verification Oracle (Phase A Complexity-Aware Scaling + Phase B)
            metrics.record_oracle_attempt()
            ver_t0 = time.perf_counter()
            ver_res = oracle.verify(
                full_code=full_code,
                seed=b_seed_val,
                prompt_hash=prompt_hash,
                generator_model=model_name,
                solve_opt=blocks["solve_opt"],
                solve_brute=blocks["solve_brute"],
                test_gen=blocks["test_gen"],
            )
            metrics.record_verification_latency(time.perf_counter() - ver_t0)

            if ver_res["phase_a_pass"]:
                metrics.record_phase_a_pass()
            else:
                metrics.record_phase_a_failure()

            if ver_res["phase_b_pass"]:
                metrics.record_phase_b_pass()
            elif ver_res["phase_a_pass"]:
                metrics.record_phase_b_failure()

            if not ver_res["verified"]:
                print(
                    f"  [X] Verification Failed: {ver_res['reason']} "
                    f"(Quarantine ID: {ver_res['quarantine_id']})"
                )
                continue

            # 4. Structural Deduplication
            problem_id = f"PROB_UNI_{b_seed_val}"
            dedup_res = dedup.register_candidate(problem_id, full_code)
            if dedup_res.is_duplicate:
                print(
                    f"  [X] Deduplication Dropped: Similarity "
                    f"{dedup_res.similarity_score:.3f} >= 0.70 to "
                    f"{dedup_res.nearest_neighbor_problem}"
                )
                metrics.record_dedup_rejection()
                continue

            # 3. Empirical Difficulty Calibration
            pass1, pass10, diff = calibrate_empirical_difficulty(full_code, b_seed_val)

            # 4. Record in Ledger
            entry = LedgerEntry(
                problem_id=problem_id,
                seed=b_seed_val,
                generator_model=model_name,
                generator_version="v1.0",
                prompt_hash=prompt_hash,
                target_failure_cluster="UNIFORM_CONTROL",
                monotonic_queue_subtype=cluster_id,
                base_pass1=pass1,
                base_pass10=pass10,
                empirical_difficulty=diff,
                oracle_confidence=ver_res["confidence"],
                phase_a_pass_count=ver_res["phase_a_pass_count"],
                phase_b_pass=ver_res["phase_b_pass"],
                verification_status="APPROVED",
                ast_hash=dedup_res.ast_hash,
                cfg_hash=dedup_res.cfg_hash,
                similarity_score=dedup_res.similarity_score,
                acceptance_reason="Approved uniform control candidate",
                code_content=full_code,
                spec=blocks["spec"],
            )
            ledger.record_entry(entry)
            accepted += 1
            metrics.record_approved(cluster=cluster_id)
            print(
                f"  [✓] ACCEPTED & RECORDED (CONTROL): {problem_id} "
                f"(Diff: {diff:.2f}, AST: {dedup_res.ast_hash})"
            )

    # Flush metrics on completion
    metrics.flush(
        json_path="data/metrics_uniform.json",
        csv_path="data/metrics_uniform.csv",
    )
    print("\n[METRICS] Flushed to data/metrics_uniform.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Uniform Control Curriculum Generation")
    parser.add_argument("--num-samples", type=int, default=2)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of parallel generation calls per outer iteration (default: 1 = serial).",
    )
    args = parser.parse_args()

    run_uniform_generation_pipeline(
        num_samples=args.num_samples,
        batch_size=args.batch_size,
    )
