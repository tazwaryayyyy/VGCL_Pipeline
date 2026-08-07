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
from typing import Any, Dict, List, Tuple
from dotenv import load_dotenv

from cfg_dedup import StructuralDeduplicationEngine
from experiment_ledger import ExperimentLedger, LedgerEntry
from oracle_refactor import VerificationOracle
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
) -> None:
    print("=== STARTING UNIFORM SAMPLING CONTROL GENERATION PIPELINE ===")

    oracle = VerificationOracle(quarantine_dir="data/quarantine_uniform")
    dedup = StructuralDeduplicationEngine(similarity_threshold=0.70)
    ledger = ExperimentLedger(db_path=db_path, jsonl_path=jsonl_path)

    seed = 5000
    accepted = 0

    while accepted < num_samples:
        cluster_id = UNIFORM_CLUSTERS[accepted % len(UNIFORM_CLUSTERS)]
        seed += 1

        print(f"\n[+] Generating Uniform Sample [{accepted + 1}/{num_samples}] | Category: '{cluster_id}' | Seed: {seed}")

        ok, blocks, model_name, latency, retries = generate_uniform_candidate(
            cluster_id=cluster_id,
            seed=seed,
        )

        if not ok:
            print("  [X] Decoupled Generation Error: Block parsing failed.")
            continue

        full_code = f"{blocks['solve_opt']}\n\n{blocks['solve_brute']}\n\n{blocks['test_gen']}"
        prompt_hash = hashlib.sha256(blocks['spec'].encode("utf-8")).hexdigest()[:16]

        # 1. Verification Oracle
        ver_res = oracle.verify(
            full_code=full_code,
            seed=seed,
            prompt_hash=prompt_hash,
            generator_model=model_name,
            solve_opt=blocks["solve_opt"],
            solve_brute=blocks["solve_brute"],
            test_gen=blocks["test_gen"],
        )

        if not ver_res["verified"]:
            print(f"  [X] Verification Failed: {ver_res['reason']} (Quarantine ID: {ver_res['quarantine_id']})")
            continue

        # 2. Structural Deduplication
        problem_id = f"PROB_UNI_{seed}"
        dedup_res = dedup.register_candidate(problem_id, full_code)
        if dedup_res.is_duplicate:
            print(f"  [X] Deduplication Dropped: Similarity {dedup_res.similarity_score:.3f} >= 0.70 to {dedup_res.nearest_neighbor_problem}")
            continue

        # 3. Empirical Difficulty Calibration
        pass1, pass10, diff = calibrate_empirical_difficulty(full_code, seed)

        # 4. Record in Ledger
        entry = LedgerEntry(
            problem_id=problem_id,
            seed=seed,
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
        print(f"  [✓] ACCEPTED & RECORDED (CONTROL): {problem_id} (Diff: {diff:.2f}, AST: {dedup_res.ast_hash})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Uniform Control Curriculum Generation")
    parser.add_argument("--num-samples", type=int, default=2)
    args = parser.parse_args()

    run_uniform_generation_pipeline(num_samples=args.num_samples)
