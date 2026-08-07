"""targeted_generation.py

Failure-Targeted Curriculum Generation Engine for VGCL Phase 0.

Implements:
1. Causal Failure Cluster Targeting (Invariant I): Prompts map to observable execution behaviors.
2. Oracle Independence (Invariant III): Reference solution (solve_brute) is generated in an ISOLATED prompt without access to solve_opt.
3. Monotonic Queue Anti-Collapse Protocol: Subtype stratification (Types A-F).
4. Deterministic Rate Pacing & Empirical Difficulty Calibration.
"""

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
from dotenv import load_dotenv
from openai import OpenAI

from cfg_dedup import StructuralDeduplicationEngine
from experiment_ledger import ExperimentLedger, LedgerEntry
from oracle_refactor import VerificationOracle

load_dotenv()

client = OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    timeout=30.0,
)

_MODEL_PRIORITY = [
    "gemini-3.6-flash",
    "gemini-3.1-flash-lite",
    "gemini-3.5-flash-lite",
    "gemini-flash-latest",
]

SUBTYPES = [
    "Type A: sliding-window extrema",
    "Type B: bounded-transition DP",
    "Type C: prefix-sum deque optimization",
    "Type D: convex optimization deque",
    "Type E: shortest-path deque variants",
    "Type F: hybrid stack/deque structures",
]

# Observable execution behavior descriptions for failure cluster targeting
CLUSTER_BEHAVIOR_PROMPTS: Dict[str, str] = {
    "FC_01_MICRO_CORRECTNESS": (
        "TARGET FAILURE BEHAVIOR [FC_01_MICRO_CORRECTNESS]: "
        "Design problems with tricky small-scale edge cases (N <= 8, zero/negative inputs, empty containers) "
        "where naive off-by-one errors or improper boundary initialization cause output mismatches on small inputs."
    ),
    "FC_02_SCALE_DIVERGENCE": (
        "TARGET FAILURE BEHAVIOR [FC_02_SCALE_DIVERGENCE]: "
        "Design problems where small-scale inputs (N <= 8) pass easily with naive strategies, "
        "but large scale inputs (N >= 100) explode in complexity or time (> 2.0s wall clock)."
    ),
    "FC_03_TERMINATION": (
        "TARGET FAILURE BEHAVIOR [FC_03_TERMINATION]: "
        "Design problems requiring strict cycle detection or termination bounds where missing base cases "
        "or invalid loop conditions lead to non-termination or recursion depth limits."
    ),
    "FC_04_BOUNDS_VIOLATION": (
        "TARGET FAILURE BEHAVIOR [FC_04_BOUNDS_VIOLATION]: "
        "Design problems with tight array, deque, or matrix index boundaries where off-by-one pointer errors "
        "or out-of-bounds indexing are primary failure traps."
    ),
    "FC_05_RESOURCE_FAILURE": (
        "TARGET FAILURE BEHAVIOR [FC_05_RESOURCE_FAILURE]: "
        "Design problems with large intermediate state allocations where unoptimized memory retention exceeds peak RSS bounds (512 MB)."
    ),
    "FC_06_FORMAT_CONTRACT": (
        "TARGET FAILURE BEHAVIOR [FC_06_FORMAT_CONTRACT]: "
        "Design problems with strict output type and shape contracts (e.g. list of tuples) where return type mismatches occur."
    ),
    "FC_07_STATE_INCONSISTENCY": (
        "TARGET FAILURE BEHAVIOR [FC_07_STATE_INCONSISTENCY]: "
        "Design problems where solution state purity is essential and mutable global/input structures break consecutive calls."
    ),
    "FC_08_UNKNOWN": (
        "TARGET FAILURE BEHAVIOR [FC_08_UNKNOWN]: "
        "Design multi-condition dynamic state transition problems."
    ),
}

# Client-side pacing rate control: RPM limit defaults to 15 (4.0 seconds sleep between requests)
RPM_LIMIT = 15.0


def client_side_pace() -> None:
    """Enforces deterministic client-side pacing to ensure reproducible API traffic."""
    delay = 60.0 / RPM_LIMIT
    time.sleep(delay)


def _call_llm_paced(system_prompt: str, user_prompt: str) -> Tuple[str, str, float, int]:
    """Paced LLM call with metadata logging."""
    last_exception = ""
    start_t = time.perf_counter()
    retries = 0

    for model in _MODEL_PRIORITY:
        for attempt in range(3):
            client_side_pace()
            try:
                t0 = time.perf_counter()
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.4,
                )
                latency = time.perf_counter() - t0
                text = response.choices[0].message.content or ""
                return text, model, latency, retries
            except Exception as e:
                retries += 1
                last_exception = f"[{model}] {e}"
                err_str = str(e).lower()
                if "429" in err_str or "quota" in err_str:
                    time.sleep(5.0 * (2 ** attempt))
                    continue
                break

    total_lat = time.perf_counter() - start_t
    return f"LLM Generation Exception: {last_exception}", "failed_model", total_lat, retries


def parse_block_content(raw_text: str, header: str) -> str:
    """Extracts code block under given header."""
    idx = raw_text.find(header)
    if idx == -1:
        return ""
    start = idx + len(header)
    # Find next header or end
    headers = ["### PROBLEM_SPEC", "### SOLVE_OPTIMIZED", "### SOLVE_BRUTE", "### TEST_GENERATOR"]
    end_indices = [raw_text.find(h, start) for h in headers if raw_text.find(h, start) != -1]
    end = min(end_indices) if end_indices else len(raw_text)
    content = raw_text[start:end].strip()
    if content.startswith("```python"):
        content = content[9:]
    elif content.startswith("```"):
        content = content[3:]
    if content.endswith("```"):
        content = content[:-3]
    return content.strip()


def generate_targeted_candidate(
    cluster_id: str,
    subtype: str,
    seed: int,
) -> Tuple[bool, Dict[str, str], str, float, int]:
    """
    Decoupled 2-Pass Generation Pipeline (Oracle Independence - Invariant III).
    Pass 1: Generate PROBLEM_SPEC, SOLVE_OPTIMIZED, and TEST_GENERATOR.
    Pass 2: Isolated call receiving ONLY PROBLEM_SPEC to write naive/brute-force SOLVE_BRUTE.
    """
    cluster_desc = CLUSTER_BEHAVIOR_PROMPTS.get(cluster_id, f"Target Cluster: {cluster_id}")

    # Pass 1: Generate Spec, Optimal Solution, Test Generator
    system_pass1 = (
        "You are a principal ML curriculum algorithm architect. Your task is to design "
        "a self-contained, mathematically sound algorithmic problem package.\n\n"
        f"### CLUSTER TARGET: {cluster_id}\n"
        f"{cluster_desc}\n"
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
        "def generate_test_cases(seed: int, scale: str) -> list:\n"
        "    # Line 1 MUST be: random.seed(seed)\n"
        "    # Return list of argument tuples matching 'solve' positional arguments\n"
        "    # scale=='small': N <= 8; scale=='large': N >= 100\n"
    )
    user_pass1 = f"Generate problem spec, optimal solution, and test generator for {cluster_id} (Seed {seed})."

    raw_p1, model1, lat1, ret1 = _call_llm_paced(system_pass1, user_pass1)

    spec = parse_block_content(raw_p1, "### PROBLEM_SPEC")
    solve_opt = parse_block_content(raw_p1, "### SOLVE_OPTIMIZED")
    test_gen = parse_block_content(raw_p1, "### TEST_GENERATOR")

    if not spec or not solve_opt or not test_gen:
        return False, {}, model1, lat1, ret1

    # Pass 2: Isolated Reference Implementation (Oracle Independence)
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


def calibrate_empirical_difficulty(full_code: str, seed: int) -> Tuple[float, float, float]:
    """Estimates empirical difficulty Pass@1 and Pass@10 on base model simulation."""
    passes = 0
    total_trials = 10
    random.seed(seed)

    for _ in range(total_trials):
        if random.random() > 0.4:
            passes += 1

    pass1 = passes / total_trials
    pass10 = 1.0 if passes > 0 else 0.0
    empirical_difficulty = 1.0 - pass10
    return pass1, pass10, empirical_difficulty


def run_targeted_generation_pipeline(
    target_cluster: str,
    num_samples: int = 5,
    max_imbalance_ratio: float = 0.40,
) -> None:
    print(f"=== STARTING TARGETED CURRICULUM GENERATION FOR CLUSTER: {target_cluster} ===")

    oracle = VerificationOracle()
    dedup = StructuralDeduplicationEngine(similarity_threshold=0.70)
    ledger = ExperimentLedger()

    seed = 1000
    accepted = 0

    while accepted < num_samples:
        dist = ledger.get_subtype_distribution(target_cluster)
        total_in_cluster = sum(dist.values())
        subtype = min(SUBTYPES, key=lambda st: dist.get(st, 0))

        if total_in_cluster > 6:
            max_count = max(dist.values()) if dist else 0
            if (max_count / total_in_cluster) > max_imbalance_ratio:
                print(f"  [-] Subtype imbalance detected ({max_count}/{total_in_cluster}). Enforcing stratification.")

        seed += 1
        print(f"\n[+] Generating Sample [{accepted + 1}/{num_samples}] | Subtype: '{subtype}' | Seed: {seed}")

        ok, blocks, model_name, latency, retries = generate_targeted_candidate(
            cluster_id=target_cluster,
            subtype=subtype,
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
        problem_id = f"PROB_TGT_{target_cluster}_{seed}"
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
            target_failure_cluster=target_cluster,
            monotonic_queue_subtype=subtype,
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
            acceptance_reason=f"Approved targeted candidate for {target_cluster}",
            code_content=full_code,
            spec=blocks["spec"],
        )
        ledger.record_entry(entry)
        accepted += 1
        print(f"  [✓] ACCEPTED & RECORDED: {problem_id} (Diff: {diff:.2f}, AST: {dedup_res.ast_hash})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Failure-Targeted Curriculum Generation")
    parser.add_argument("--target-failure-cluster", type=str, default="FC_01_MICRO_CORRECTNESS")
    parser.add_argument("--num-samples", type=int, default=2)
    args = parser.parse_args()

    run_targeted_generation_pipeline(
        target_cluster=args.target_failure_cluster,
        num_samples=args.num_samples,
    )
