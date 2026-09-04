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
from generation_manifest import create_generation_manifest
from targeted_generation import (
    EXPERIMENTAL_GENERATOR_POLICY,
    SUBTYPES,
    _MODEL_PRIORITY,
    _call_llm_paced,
    _get_prober,
    client_side_pace,
    parse_block_content,
)
from base_model_prober import BaseModelProber
from quality_gates import check_primary_uniform_eligibility


load_dotenv()


def generate_uniform_candidate(
    subtype: str,
    seed: int,
    metrics: Optional[PipelineMetrics] = None,
) -> Tuple[bool, Dict[str, str], str, str, float, int, bool, bool, str, str]:
    """
    Decoupled 2-Pass Generation Pipeline (Control Condition - Invariant V).
    Pass 1: Generate PROBLEM_SPEC, SOLVE_OPTIMIZED, and TEST_GENERATOR uniformly from the shared
    support domain (Monotonic Queue / Deque) without failure targeting or cluster descriptions.
    Pass 2: Isolated call receiving ONLY PROBLEM_SPEC to write naive/brute-force SOLVE_BRUTE.

    Returns:
        (ok, blocks, model1, model2, latency, api_calls, fb1, fb2, fb_reason1, fb_reason2)
    """
    system_pass1 = (
        "You are a principal competitive programming problem architect. Design a self-contained, "
        "mathematically sound algorithmic problem package.\n\n"
        f"Algorithmic Domain: Monotonic Queue / Deque Structures\n"
        f"Subtype Stratification: {subtype}\n\n"
        "### OUTPUT FORMAT CONTRACT (Pass 1):\n"
        "Output EXACTLY three headers:\n\n"
        "### PROBLEM_SPEC\n"
        "<Problem description, input/output formats, scale constraints>\n\n"
        "### SOLVE_OPTIMIZED\n"
        "def solve(...):\n"
        "    <Optimal implementation>\n\n"
        "### TEST_GENERATOR\n"
        "import random\n"
        "def generate_test_cases(seed: int, scale: str, max_small_dimension: int = 8) -> list:\n"
        "    # Line 1 MUST be: random.seed(seed)\n"
        "    # Return list of argument tuples matching 'solve' positional arguments\n"
        "    # scale=='small': dimensions <= max_small_dimension (default 8, or bounded cap if specified);\n"
        "    # scale=='large': derive dimensions from declared problem scale constraints (e.g. N >= 100)\n"
    )
    user_pass1 = f"Generate problem spec, optimal solution, and test generator for subtype '{subtype}' (Seed {seed})."

    raw_p1, model1, lat1, ret1, fb1, fb_reason1 = _call_llm_paced(system_pass1, user_pass1, metrics=metrics)

    spec = parse_block_content(raw_p1, "### PROBLEM_SPEC")
    solve_opt = parse_block_content(raw_p1, "### SOLVE_OPTIMIZED")
    test_gen = parse_block_content(raw_p1, "### TEST_GENERATOR")

    if not spec or not solve_opt or not test_gen:
        return False, {}, model1, "", lat1, ret1, fb1, False, fb_reason1, ""

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

    raw_p2, model2, lat2, ret2, fb2, fb_reason2 = _call_llm_paced(system_pass2, user_pass2, metrics=metrics)
    solve_brute = parse_block_content(raw_p2, "### SOLVE_BRUTE")

    if not solve_brute:
        return False, {}, model1, model2, lat1 + lat2, ret1 + ret2, fb1, fb2, fb_reason1, fb_reason2

    blocks = {
        "spec": spec,
        "solve_opt": solve_opt,
        "solve_brute": solve_brute,
        "test_gen": test_gen,
    }
    return True, blocks, model1, model2, lat1 + lat2, ret1 + ret2, fb1, fb2, fb_reason1, fb_reason2


def run_uniform_generation_pipeline(
    num_samples: int = 5,
    db_path: str = "data/v7_uniform_experiment_ledger.db",
    jsonl_path: str = "data/v7_uniform_approved_curriculum.jsonl",
    aux_jsonl_path: str = "data/v7_uniform_auxiliary_curriculum.jsonl",
    batch_size: int = 1,
    metrics: Optional[PipelineMetrics] = None,
) -> None:
    """
    Uniform sampling control generation loop with shared domain support and V7 primary gating.

    Args:
        num_samples:    Number of verified, deduplicated primary samples to accept.
        db_path:        SQLite ledger path for the uniform condition.
        jsonl_path:     JSONL output path for the primary uniform condition.
        aux_jsonl_path: JSONL output path for auxiliary uniform records.
        batch_size:     Number of parallel generation calls per outer iteration.
        metrics:        Optional PipelineMetrics instance.
    """
    print("=== STARTING UNIFORM SAMPLING CONTROL GENERATION PIPELINE ===")
    manifest = create_generation_manifest(output_path="data/generation_manifest_uniform.json")
    print(
        f"  [MANIFEST] Frozen manifest recorded (Freeze: {manifest.get('freeze_tag')}, "
        f"Primary Model: {manifest.get('generator_model')})"
    )

    oracle = VerificationOracle(quarantine_dir="data/quarantine_uniform")
    dedup = StructuralDeduplicationEngine(similarity_threshold=0.70)
    ledger = ExperimentLedger(db_path=db_path, jsonl_path=jsonl_path, aux_jsonl_path=aux_jsonl_path)

    if metrics is None:
        metrics = PipelineMetrics()

    seed = 5000
    accepted = 0

    while accepted < num_samples:
        # Pre-compute seeds for all items in this batch before dispatch
        batch_seeds = [seed + i + 1 for i in range(batch_size)]
        seed += batch_size

        # Subtypes for this batch sampled round-robin across shared domain support
        batch_subtypes = [
            SUBTYPES[(accepted + i) % len(SUBTYPES)]
            for i in range(batch_size)
        ]

        def _generate_one_uniform(
            subtype: str,
            b_seed: int,
        ) -> Tuple[int, str, bool, Dict[str, str], str, str, float, int, bool, bool, str, str]:
            """Worker thunk executed in the thread pool."""
            ok, blocks, m1, m2, latency, retries, fb1, fb2, fb_r1, fb_r2 = generate_uniform_candidate(
                subtype=subtype,
                seed=b_seed,
                metrics=metrics,
            )
            return b_seed, subtype, ok, blocks, m1, m2, latency, retries, fb1, fb2, fb_r1, fb_r2

        effective_batch = min(batch_size, num_samples - accepted)
        futures_map = {}
        with ThreadPoolExecutor(max_workers=effective_batch) as executor:
            for i in range(effective_batch):
                fut = executor.submit(
                    _generate_one_uniform,
                    batch_subtypes[i],
                    batch_seeds[i],
                )
                futures_map[fut] = (batch_seeds[i], batch_subtypes[i])

            results_by_seed: Dict[int, Any] = {}
            for fut in as_completed(futures_map):
                b_seed, subtype = futures_map[fut]
                try:
                    result = fut.result()
                except Exception as exc:
                    result = (b_seed, subtype, False, {}, "failed", "failed", 0.0, 0, False, False, "", "")
                    print(f"  [X] Batch worker exception (seed {b_seed}): {exc}")
                results_by_seed[b_seed] = result

        # Process batch results in deterministic seed order
        for b_seed in sorted(results_by_seed.keys()):
            if accepted >= num_samples:
                break

            b_seed_val, subtype, ok, blocks, m1, m2, latency, api_calls, fb1, fb2, fb_r1, fb_r2 = results_by_seed[b_seed]

            metrics.record_attempt(cluster="UNIFORM_CONTROL")
            metrics.record_gen_latency(latency)
            metrics.record_initial_generation_api_calls(api_calls)
            metrics.record_model_serving(m1)

            gen_fallback_used = fb1 or fb2
            gen_fallback_reason = f"Pass1: {fb_r1} | Pass2: {fb_r2}".strip(" |") if gen_fallback_used else ""
            primary_model_name = EXPERIMENTAL_GENERATOR_POLICY["primary_model"]
            effective_model = primary_model_name if not gen_fallback_used else (f"{m1}+{m2}" if m1 != m2 else m1)

            print(
                f"\n[+] Generating Uniform Sample [{accepted + 1}/{num_samples}] "
                f"| Subtype: '{subtype}' | Seed: {b_seed_val}"
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
                        cluster_id="UNIFORM_CONTROL",
                        subtype=subtype,
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
                        generator_model=effective_model,
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
                    generator_model=effective_model,
                    generator_version="v1.0",
                    optimized_code=blocks.get("solve_opt", ""),
                    reference_code=blocks.get("solve_brute", ""),
                    test_generator_code=blocks.get("test_gen", ""),
                    quarantine_reason=f"Reference Independence Violation: {ref_ind['reason']} (Score: {ref_ind['similarity_score']})",
                    execution_traces={"reference_independence": ref_ind},
                )
                metrics.record_reference_independence_rejection()
                continue

            # 2. Subtype Validity Gate (validates that uniform sample matches shared domain subtype)
            st_val = validate_subtype(subtype, blocks["spec"], blocks["solve_opt"])
            if st_val["status"] == "REJECT":
                print(f"  [X] Subtype Validation Failed for '{subtype}': {st_val['reason']}")
                oracle.quarantine(
                    seed=b_seed_val,
                    prompt_hash=prompt_hash,
                    generator_model=effective_model,
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
                generator_model=effective_model,
                solve_opt=blocks["solve_opt"],
                solve_brute=blocks["solve_brute"],
                test_gen=blocks["test_gen"],
                spec=blocks["spec"],
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

            # 5. Real Base-Model Probing (V7: Execution against frozen Qwen base model for difficulty)
            prober = _get_prober()
            probe_result = prober.probe(
                problem_id=problem_id,
                problem_spec=blocks["spec"],
                solve_brute=blocks["solve_brute"],
                test_gen=blocks["test_gen"],
                seed=b_seed_val,
            )
            pass1 = probe_result.pass1
            pass10 = probe_result.pass10
            diff = probe_result.empirical_difficulty

            probe_dict_json = json.dumps(probe_result.to_dict(), sort_keys=True)
            probe_hash = hashlib.sha256(probe_dict_json.encode()).hexdigest()[:16]
            target_evidence_json = json.dumps(probe_result.failure_evidence)

            # 6. Construct Ledger Entry
            entry = LedgerEntry(
                problem_id=problem_id,
                seed=b_seed_val,
                generator_model=effective_model,
                generator_version="v1.0",
                prompt_hash=prompt_hash,
                target_failure_cluster="UNIFORM_CONTROL",
                monotonic_queue_subtype=subtype,
                # Target attribution (NOT_APPLICABLE for uniform control)
                target_validation_status="NOT_APPLICABLE",
                target_evidence=target_evidence_json,
                probe_result_hash=probe_hash,
                # Generator policy fields
                generator_policy="EXPERIMENTAL_GENERATOR_POLICY",
                generator_model_pass1=m1,
                generator_model_pass2=m2,
                fallback_used=gen_fallback_used,
                fallback_used_pass1=fb1,
                fallback_used_pass2=fb2,
                fallback_reason=gen_fallback_reason,
                fallback_reason_pass1=fb_r1,
                fallback_reason_pass2=fb_r2,
                # Difficulty from real probe
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
                acceptance_reason="Approved uniform control candidate [target_validation=NOT_APPLICABLE]",
                code_content=full_code,
                spec=blocks["spec"],
            )

            # 7. V7 Authoritative Primary Eligibility Gate (Uniform)
            eligibility = check_primary_uniform_eligibility(entry)
            entry.primary_eligibility_status = eligibility["status"]
            entry.primary_eligibility_reason = (
                "; ".join(eligibility["reasons"]) if eligibility["reasons"] else "Meets all primary invariants"
            )

            ledger.record_entry(entry)

            if eligibility["is_primary"]:
                accepted += 1
                metrics.record_approved(cluster="UNIFORM_CONTROL")
                print(
                    f"  [✓] ACCEPTED AS PRIMARY (CONTROL) [{accepted}/{num_samples}]: {problem_id} "
                    f"(Diff: {diff:.2f}, AST: {dedup_res.ast_hash})"
                )
            else:
                print(
                    f"  [-] NON-PRIMARY / AUXILIARY ({entry.primary_eligibility_status}): {problem_id} "
                    f"— Reason: {entry.primary_eligibility_reason}"
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
