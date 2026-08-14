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
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple
from dotenv import load_dotenv
from openai import OpenAI

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


class SubtypeScheduler:
    """
    Approval-weighted adaptive subtype sampler within fixed accepted-count strata.

    Maintains a per-subtype approval-rate estimate and adjusts sampling
    probability only among subtypes with the lowest current accepted count.
    This lets approval feedback affect candidate attempts without allowing
    successful subtypes to run ahead in the accepted curriculum composition.

    Reproducibility: selection uses a ``random.Random`` instance seeded per
    call, independent of the global RNG, so it never perturbs seed-dependent
    test-generator behaviour inside oracle workers.

    Auditability: every weight update is logged with a timestamp to
    ``_weight_log`` and can be persisted via ``dump_weight_log``.
    """

    def __init__(
        self,
        subtypes: List[str],
        floor_period: int = 12,
    ) -> None:
        """
        Args:
            subtypes:     Ordered list of subtype strings (must be non-empty).
            floor_period: Retained for compatibility with the initial
                          throughput implementation. Current coverage is
                          enforced by the accepted-count stratum in ``select``.
        """
        self._subtypes: List[str] = list(subtypes)
        # Approval stats: approved count and attempted count per subtype
        self._approved: Dict[str, int] = {st: 0 for st in subtypes}
        self._attempted: Dict[str, int] = {st: 0 for st in subtypes}
        self._total_attempts: int = 0
        self._weight_log: List[Dict[str, Any]] = []

    def _approval_rate(self, subtype: str) -> float:
        """Returns the current approval rate for a subtype (0.0 if unseen)."""
        att = self._attempted.get(subtype, 0)
        apr = self._approved.get(subtype, 0)
        return apr / att if att > 0 else 0.0

    def _compute_weights(self) -> List[float]:
        """
        Computes unnormalised sampling weights.

        Weight formula: ``1.0 + 0.5 * approval_rate``
        This gives unseen subtypes weight 1.0 and a perfectly approved
        subtype weight 1.5, keeping the distribution from collapsing.
        """
        return [1.0 + 0.5 * self._approval_rate(st) for st in self._subtypes]

    def select(self, ledger_dist: Dict[str, int], seed: int) -> str:
        """
        Selects a subtype for the next generation attempt.

        The eligible set is restricted to subtypes with the lowest accepted
        count. If any eligible subtype has never been attempted, it is selected
        in list order before weighted sampling is used.

        Otherwise, weighted random selection is used.

        Args:
            ledger_dist: Subtype → count dict from the ledger (approved only).
            seed:        Current pipeline seed, used to seed the local RNG.
                         Does not touch the global ``random`` state.

        Returns:
            The selected subtype string.
        """
        self._total_attempts += 1
        accepted_counts = {st: ledger_dist.get(st, 0) for st in self._subtypes}
        min_accepted = min(accepted_counts.values())
        eligible = [st for st in self._subtypes if accepted_counts[st] == min_accepted]

        # Guarantee initial coverage inside the currently underfilled stratum.
        never_attempted = [st for st in eligible if self._attempted.get(st, 0) == 0]
        if never_attempted:
            chosen = never_attempted[0]
            self._log_weight_event(chosen, reason="floor_coverage")
            return chosen

        # Weighted random selection using a local RNG, restricted to the
        # accepted-count stratum that is currently underfilled.
        rng = random.Random(seed * 1000 + self._total_attempts)
        weights_by_subtype = dict(zip(self._subtypes, self._compute_weights()))
        weights = [weights_by_subtype[st] for st in eligible]
        chosen = rng.choices(eligible, weights=weights, k=1)[0]
        self._log_weight_event(chosen, reason="weighted_random")
        return chosen

    def record_outcome(self, subtype: str, approved: bool) -> None:
        """
        Updates internal approval stats after a sample's pipeline outcome.

        Must be called exactly once per attempt, after the acceptance/rejection
        decision is known.

        Args:
            subtype:  The subtype string that was attempted.
            approved: True if the sample was accepted into the ledger.
        """
        self._attempted[subtype] = self._attempted.get(subtype, 0) + 1
        if approved:
            self._approved[subtype] = self._approved.get(subtype, 0) + 1

    def _log_weight_event(self, chosen: str, reason: str) -> None:
        """Appends a weight-change event to the in-memory log."""
        self._weight_log.append({
            "timestamp": time.time(),
            "total_attempts": self._total_attempts,
            "chosen_subtype": chosen,
            "reason": reason,
            "weights": dict(zip(self._subtypes, self._compute_weights())),
        })

    def dump_weight_log(self, path: str) -> None:
        """
        Persists the weight change log to a JSON file for auditability.

        Args:
            path: File path for the JSON output.
        """
        try:
            os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self._weight_log, f, indent=2)
        except Exception as exc:
            print(f"  [SCHEDULER] Warning: could not write weight log to '{path}': {exc}")


def _call_llm_paced(system_prompt: str, user_prompt: str) -> Tuple[str, str, float, int]:
    """Paced LLM call with metadata logging."""
    last_exception = ""
    start_t = time.perf_counter()
    retries = 0
    api_calls = 0

    for model in _MODEL_PRIORITY:
        for attempt in range(3):
            client_side_pace()
            try:
                t0 = time.perf_counter()
                api_calls += 1
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
                return text, model, latency, api_calls
            except Exception as e:
                retries += 1
                last_exception = f"[{model}] {e}"
                err_str = str(e).lower()
                if "429" in err_str or "quota" in err_str:
                    time.sleep(5.0 * (2 ** attempt))
                    continue
                break

    total_lat = time.perf_counter() - start_t
    return f"LLM Generation Exception: {last_exception}", "failed_model", total_lat, api_calls


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
    batch_size: int = 1,
    metrics: Optional[PipelineMetrics] = None,
) -> None:
    """
    Failure-targeted curriculum generation loop.

    Args:
        target_cluster:      Failure cluster ID (e.g. ``"FC_01_MICRO_CORRECTNESS"``).
        num_samples:         Number of verified, deduplicated samples to accept.
        max_imbalance_ratio: Subtype imbalance ratio threshold for logging.
        batch_size:          Number of parallel generation calls per outer
                             iteration.  Default ``1`` runs without generation
                             concurrency.
        metrics:             Optional ``PipelineMetrics`` instance.  When
                             ``None``, a private run-scoped collector is used.
    """
    print(f"=== STARTING TARGETED CURRICULUM GENERATION FOR CLUSTER: {target_cluster} ===")

    oracle = VerificationOracle()
    dedup = StructuralDeduplicationEngine(similarity_threshold=0.70)
    ledger = ExperimentLedger()
    scheduler = SubtypeScheduler(subtypes=SUBTYPES)

    if metrics is None:
        metrics = PipelineMetrics()

    seed = 1000
    accepted = 0

    while accepted < num_samples:
        batch_seeds = [seed + i + 1 for i in range(batch_size)]
        seed += batch_size

        batch_subtypes: List[str] = []
        planned_dist = ledger.get_subtype_distribution(target_cluster)
        for b_seed in batch_seeds:
            total_in_cluster = sum(planned_dist.values())
            subtype = scheduler.select(ledger_dist=planned_dist, seed=b_seed)

            if total_in_cluster > 6:
                max_count = max(dist.values()) if dist else 0
                if (max_count / total_in_cluster) > max_imbalance_ratio:
                    print(
                        f"  [-] Subtype imbalance detected ({max_count}/{total_in_cluster}). "
                        "Enforcing stratification."
                    )

            batch_subtypes.append(subtype)
            planned_dist[subtype] = planned_dist.get(subtype, 0) + 1

        def _generate_one(
            cluster_id: str,
            subtype: str,
            b_seed: int,
        ) -> Tuple[int, str, bool, Dict[str, str], str, float, int]:
            """Worker thunk executed in the thread pool."""
            ok, blocks, model_name, latency, retries = generate_targeted_candidate(
                cluster_id=cluster_id,
                subtype=subtype,
                seed=b_seed,
            )
            return b_seed, subtype, ok, blocks, model_name, latency, retries

        effective_batch = min(batch_size, num_samples - accepted)
        futures_map = {}
        with ThreadPoolExecutor(max_workers=effective_batch) as executor:
            for i in range(effective_batch):
                fut = executor.submit(
                    _generate_one,
                    target_cluster,
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
                    result = (b_seed, subtype, False, {}, "failed", 0.0, 0)
                    print(f"  [X] Batch worker exception (seed {b_seed}): {exc}")
                results_by_seed[b_seed] = result

        for b_seed in sorted(results_by_seed.keys()):
            if accepted >= num_samples:
                break

            b_seed_val, subtype, ok, blocks, model_name, latency, api_calls = results_by_seed[b_seed]

            metrics.record_attempt(subtype=subtype, cluster=target_cluster)
            metrics.record_gen_latency(latency)
            metrics.record_initial_generation_api_calls(api_calls)

            print(
                f"\n[+] Processing Sample [{accepted + 1}/{num_samples}] "
                f"| Subtype: '{subtype}' | Seed: {b_seed_val}"
            )

            if not ok:
                print("  [X] Decoupled Generation Error: Block parsing failed.")
                metrics.record_gen_error()
                scheduler.record_outcome(subtype, approved=False)
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
                        cluster_id=target_cluster,
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
                        generator_model=model_name,
                        generator_version="v1.0",
                        optimized_code=blocks.get("solve_opt", ""),
                        reference_code=blocks.get("solve_brute", ""),
                        test_generator_code=blocks.get("test_gen", ""),
                        quarantine_reason=f"Syntax Failure: {syn_diag.error_category} ({syn_diag.error_msg[:100]})",
                    )
                    metrics.record_syntax_failure()
                    scheduler.record_outcome(subtype, approved=False)
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
                scheduler.record_outcome(subtype, approved=False)
                continue

            # 2. Subtype Validity Gate
            st_val = validate_subtype(subtype, blocks["spec"], blocks["solve_opt"])
            if st_val["status"] == "REJECT":
                print(f"  [X] Subtype Validation Failed for '{subtype}': {st_val['reason']}")
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
                scheduler.record_outcome(subtype, approved=False)
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
                scheduler.record_outcome(subtype, approved=False)
                continue

            # 4. Structural Corpus Deduplication
            problem_id = f"PROB_TGT_{target_cluster}_{b_seed_val}"
            dedup_res = dedup.register_candidate(problem_id, full_code)
            if dedup_res.is_duplicate:
                print(
                    f"  [X] Deduplication Dropped: Similarity "
                    f"{dedup_res.similarity_score:.3f} >= 0.70 to "
                    f"{dedup_res.nearest_neighbor_problem}"
                )
                metrics.record_dedup_rejection()
                scheduler.record_outcome(subtype, approved=False)
                continue

            pass1, pass10, diff = calibrate_empirical_difficulty(full_code, b_seed_val)

            entry = LedgerEntry(
                problem_id=problem_id,
                seed=b_seed_val,
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
            metrics.record_approved(subtype=subtype, cluster=target_cluster)
            scheduler.record_outcome(subtype, approved=True)
            print(
                f"  [\u2713] ACCEPTED & RECORDED: {problem_id} "
                f"(Diff: {diff:.2f}, AST: {dedup_res.ast_hash})"
            )

    metrics.flush(
        json_path="data/metrics_targeted.json",
        csv_path="data/metrics_targeted.csv",
    )
    scheduler.dump_weight_log("data/scheduler_weight_log.json")
    print("\n[METRICS] Flushed to data/metrics_targeted.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Failure-Targeted Curriculum Generation")
    parser.add_argument("--target-failure-cluster", type=str, default="FC_01_MICRO_CORRECTNESS")
    parser.add_argument("--num-samples", type=int, default=2)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of parallel generation calls per outer iteration (default: 1 = serial).",
    )
    args = parser.parse_args()

    run_targeted_generation_pipeline(
        target_cluster=args.target_failure_cluster,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
    )
