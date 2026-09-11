"""targeted_generation.py

Failure-Targeted Curriculum Generation Engine for VGCL Phase 0.

Implements:
1. Causal Failure Cluster Targeting (Invariant I): Prompts map to observable execution behaviors.
2. Oracle Independence (Invariant III): Reference solution (solve_brute) is generated in an ISOLATED
   prompt without access to solve_opt.
3. Monotonic Queue Anti-Collapse Protocol: Subtype stratification (Types A-F), with a deterministic
   CLUSTER_SUBTYPE_COMPATIBILITY policy that prevents semantic drift (V6 fix).
4. Real Empirical Difficulty Calibration via BaseModelProber (V6 fix — replaces PRNG simulation).
5. Primary-only generator policy enforcement — fallback model usage is explicitly recorded in the
   ledger as fallback_used=True and never silently promoted to primary (V6 fix).
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
from pathlib import Path
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

from generation_manifest import create_generation_manifest
from base_model_prober import BaseModelProber, _PROBING_POLICY_HASH

load_dotenv()

client = OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    timeout=30.0,
)

EXPERIMENTAL_GENERATOR_POLICY: Dict[str, Any] = {
    "primary_model": "gemini-3.6-flash",
    "fallback_models": [
        "gemini-3.1-flash-lite",
        "gemini-3.5-flash-lite",
        "gemini-flash-latest",
    ],
    "temperature": 0.4,
    "timeout_seconds": 30.0,
}

_MODEL_PRIORITY: List[str] = [
    EXPERIMENTAL_GENERATOR_POLICY["primary_model"]
] + list(EXPERIMENTAL_GENERATOR_POLICY["fallback_models"])

SUBTYPES = [
    "Type A: sliding-window extrema",
    "Type B: bounded-transition DP",
    "Type C: prefix-sum deque optimization",
    "Type D: convex optimization deque",
    "Type E: shortest-path deque variants",
    "Type F: hybrid stack/deque structures",
]

# ---------------------------------------------------------------------------
# V7: Deterministic Cluster Behavioral Profiles
# ---------------------------------------------------------------------------
# Provides observable execution failure characteristics for prompt guidance
# without subjective interpretation or speculative reasoning.

CLUSTER_BEHAVIOR_PROMPTS: Dict[str, str] = {
    "FC_01_MICRO_CORRECTNESS": (
        "Observable failure profile: Off-by-one errors, edge-case index handling, "
        "or boundary condition mismatches on small-scale input domains."
    ),
    "FC_02_SCALE_DIVERGENCE": (
        "Observable failure profile: Solution passes micro/small scale tests but suffers "
        "time complexity blowup or quadratic scaling failures on large-scale instances."
    ),
    "FC_03_TERMINATION": (
        "Observable failure profile: Infinite loops, unbounded recursion depth, "
        "or iteration cycles failing termination contracts."
    ),
    "FC_04_BOUNDS_VIOLATION": (
        "Observable failure profile: IndexError, KeyError, or out-of-bounds pointer traversal "
        "when querying container elements."
    ),
    "FC_05_RESOURCE_FAILURE": (
        "Observable failure profile: Memory limit exceedance or excessive auxiliary space "
        "allocation exceeding system resource caps."
    ),
    "FC_06_FORMAT_CONTRACT": (
        "Observable failure profile: Return type mismatch, malformed output shape, "
        "or failure to meet the declared function signature specification."
    ),
    "FC_07_STATE_INCONSISTENCY": (
        "Observable failure profile: Unintended in-place mutation of input arguments or "
        "state corruption across successive query executions."
    ),
    "FC_08_UNKNOWN": (
        "Observable failure profile: General unclassified deterministic execution divergence."
    ),
}

# ---------------------------------------------------------------------------
# V6/V7: Cluster → Subtype Compatibility Policy
# ---------------------------------------------------------------------------
# This mapping is the SINGLE source of truth for which subtypes are compatible
# with which failure clusters. SubtypeScheduler.select() enforces this list
# at call time; requesting a subtype outside a cluster's allowed set raises
# ValueError (fail-closed).
#
# Rationale:
#   FC_01 (micro-correctness)   → boundary/edge-case subtypes A, B, D, F
#   FC_02 (scale divergence)    → complexity-intensive subtypes B, C, D
#   FC_03 (termination)         → recursion/cycle subtypes C, E, F
#   FC_04 (bounds violation)    → pointer/index subtypes A, D, E
#   FC_05 (resource failure)    → memory-intensive subtypes C, D, E
#   FC_06 (format contract)     → output-shape subtypes A, F
#   FC_07 (state inconsistency) → stateful subtypes B, E, F
#   FC_08 (unknown)             → all subtypes (catch-all)

CLUSTER_SUBTYPE_COMPATIBILITY: Dict[str, List[str]] = {
    "FC_01_MICRO_CORRECTNESS": [
        "Type A: sliding-window extrema",
        "Type B: bounded-transition DP",
        "Type D: convex optimization deque",
        "Type F: hybrid stack/deque structures",
    ],
    "FC_02_SCALE_DIVERGENCE": [
        "Type B: bounded-transition DP",
        "Type C: prefix-sum deque optimization",
        "Type D: convex optimization deque",
    ],
    "FC_03_TERMINATION": [
        "Type C: prefix-sum deque optimization",
        "Type E: shortest-path deque variants",
        "Type F: hybrid stack/deque structures",
    ],
    "FC_04_BOUNDS_VIOLATION": [
        "Type A: sliding-window extrema",
        "Type D: convex optimization deque",
        "Type E: shortest-path deque variants",
    ],
    "FC_05_RESOURCE_FAILURE": [
        "Type C: prefix-sum deque optimization",
        "Type D: convex optimization deque",
        "Type E: shortest-path deque variants",
    ],
    "FC_06_FORMAT_CONTRACT": [
        "Type A: sliding-window extrema",
        "Type F: hybrid stack/deque structures",
    ],
    "FC_07_STATE_INCONSISTENCY": [
        "Type B: bounded-transition DP",
        "Type E: shortest-path deque variants",
        "Type F: hybrid stack/deque structures",
    ],
    "FC_08_UNKNOWN": list(SUBTYPES),  # catch-all allows all subtypes
}

# Validate policy completeness at import time — fail loudly if a cluster
# references a subtype not in the global SUBTYPES list.
_policy_errors: List[str] = []
for _cluster, _allowed in CLUSTER_SUBTYPE_COMPATIBILITY.items():
    for _st in _allowed:
        if _st not in SUBTYPES:
            _policy_errors.append(f"{_cluster}: '{_st}' not in SUBTYPES")
if _policy_errors:
    raise ValueError(
        "CLUSTER_SUBTYPE_COMPATIBILITY references unknown subtypes: "
        + "; ".join(_policy_errors)
    )


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

    def select(self, ledger_dist: Dict[str, int], seed: int, cluster_id: Optional[str] = None) -> str:
        """
        Selects a subtype for the next generation attempt.

        The eligible set is first restricted to subtypes with the lowest accepted
        count, then intersected with CLUSTER_SUBTYPE_COMPATIBILITY[cluster_id] to
        enforce the semantic drift prevention policy (V6 fix).

        If no compatible subtype exists for the requested cluster, raises
        ValueError (fail-closed — never silently selects an incompatible subtype).

        If any eligible compatible subtype has never been attempted, it is selected
        in list order before weighted sampling is used.

        Args:
            ledger_dist: Subtype → count dict from the ledger (approved only).
            seed:        Current pipeline seed, used to seed the local RNG.
                         Does not touch the global ``random`` state.
            cluster_id:  Optional failure cluster ID.  When provided, restricts
                         selection to subtypes in CLUSTER_SUBTYPE_COMPATIBILITY.

        Returns:
            The selected subtype string.
        """
        self._total_attempts += 1
        accepted_counts = {st: ledger_dist.get(st, 0) for st in self._subtypes}
        min_accepted = min(accepted_counts.values())
        eligible = [st for st in self._subtypes if accepted_counts[st] == min_accepted]

        # V6: intersect with cluster compatibility policy (fail-closed)
        if cluster_id is not None:
            allowed = CLUSTER_SUBTYPE_COMPATIBILITY.get(cluster_id)
            if allowed is None:
                raise ValueError(
                    f"SubtypeScheduler.select: cluster_id '{cluster_id}' has no entry in "
                    "CLUSTER_SUBTYPE_COMPATIBILITY. Add it before running generation."
                )
            compatible_eligible = [st for st in eligible if st in allowed]
            if not compatible_eligible:
                # Broaden to the full allowed set (not just underfilled stratum)
                # to avoid deadlock when all compatible subtypes are ahead in count.
                compatible_eligible = [st for st in self._subtypes if st in allowed]
            if not compatible_eligible:
                raise ValueError(
                    f"SubtypeScheduler.select: no compatible subtypes found for cluster "
                    f"'{cluster_id}'. CLUSTER_SUBTYPE_COMPATIBILITY entry is {allowed}, "
                    f"but scheduler subtypes are {self._subtypes}."
                )
            eligible = compatible_eligible

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
        weights = [weights_by_subtype.get(st, 1.0) for st in eligible]
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


def classify_api_error(exc: Exception) -> Tuple[str, str]:
    """Classifies an API exception into standardized error category and reason."""
    err_str = str(exc).lower()
    exc_type = type(exc).__name__

    if "quota" in err_str or "resource_exhausted" in err_str or "quota_exceeded" in err_str:
        return "quota_exhaustion", f"quota_exhaustion: {exc_type}: {str(exc)[:120]}"
    elif "429" in err_str or "rate limit" in err_str or "ratelimit" in err_str or "too many requests" in err_str:
        return "rate_limit", f"rate_limit: {exc_type}: {str(exc)[:120]}"
    elif any(k in err_str for k in ("500", "502", "503", "504", "server unavailable", "service unavailable", "timeout", "timed out", "connection error", "connection reset", "overloaded")):
        return "transient_server_failure", f"transient_server_failure: {exc_type}: {str(exc)[:120]}"
    elif any(k in err_str for k in ("400", "401", "403", "404", "invalid_argument", "bad request", "permission_denied", "unauthorized")):
        return "request_rejection", f"request_rejection: {exc_type}: {str(exc)[:120]}"
    else:
        return "unknown", f"unknown: {exc_type}: {str(exc)[:120]}"


def _call_llm_paced(
    system_prompt: str,
    user_prompt: str,
    model_priority: Optional[List[str]] = None,
    client_override: Optional[Any] = None,
    metrics: Optional[PipelineMetrics] = None,
) -> Tuple[str, str, float, int, bool, str]:
    """Paced LLM call with metadata logging and explicit fallback routing.

    Returns:
        (text, model_name, latency, api_calls, fallback_used, fallback_reason)
        fallback_used is True if any model other than the primary was used.
    """
    last_exception = ""
    start_t = time.perf_counter()
    api_calls = 0
    cli = client_override if client_override is not None else client
    primary_model = EXPERIMENTAL_GENERATOR_POLICY["primary_model"]
    models = model_priority if model_priority is not None else _MODEL_PRIORITY
    fallback_reason = ""
    fallback_used = False

    for idx, model in enumerate(models):
        if idx > 0:
            fallback_used = True
            print(f"  [FALLBACK-ROUTING] Attempting declared fallback model [{idx}]: {model} (Reason: {fallback_reason})", flush=True)
            if metrics:
                metrics.record_fallback_event(reason=fallback_reason)
        for attempt in range(3):
            client_side_pace()
            try:
                t0 = time.perf_counter()
                api_calls += 1
                response = cli.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=EXPERIMENTAL_GENERATOR_POLICY.get("temperature", 0.4),
                )
                latency = time.perf_counter() - t0
                text = response.choices[0].message.content or ""
                return text, model, latency, api_calls, fallback_used, fallback_reason
            except Exception as e:
                err_class, fallback_reason = classify_api_error(e)
                if metrics:
                    metrics.record_api_event(err_class, reason=fallback_reason)
                last_exception = f"[{model}] {fallback_reason}"
                err_str = str(e).lower()
                if "429" in err_str or "quota" in err_str or "rate limit" in err_str or "resource_exhausted" in err_str:
                    time.sleep(5.0 * (2 ** attempt))
                    continue
                break

    total_lat = time.perf_counter() - start_t
    return f"LLM Generation Exception: {last_exception}", "failed_model", total_lat, api_calls, fallback_used, fallback_reason



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
    metrics: Optional[PipelineMetrics] = None,
) -> Tuple[bool, Dict[str, str], str, str, float, int, bool, bool, str, str]:
    """
    Decoupled 2-Pass Generation Pipeline (Oracle Independence - Invariant III).
    Pass 1: Generate PROBLEM_SPEC, SOLVE_OPTIMIZED, and TEST_GENERATOR.
    Pass 2: Isolated call receiving ONLY PROBLEM_SPEC to write naive/brute-force SOLVE_BRUTE.

    Returns:
        (ok, blocks, model1, model2, latency, api_calls, fb1, fb2, fb_reason1, fb_reason2)
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
        "def generate_test_cases(seed: int, scale: str, max_small_dimension: int = 8) -> list:\n"
        "    # Line 1 MUST be: random.seed(seed)\n"
        "    # Return list of argument tuples matching 'solve' positional arguments\n"
        "    # scale=='small': dimensions <= max_small_dimension (default 8, or bounded cap if specified);\n"
        "    # scale=='large': derive dimensions from declared problem scale constraints (e.g. N >= 100)\n"
    )
    user_pass1 = f"Generate problem spec, optimal solution, and test generator for {cluster_id} (Seed {seed})."

    raw_p1, model1, lat1, ret1, fb1, fb_reason1 = _call_llm_paced(system_pass1, user_pass1, metrics=metrics)

    spec = parse_block_content(raw_p1, "### PROBLEM_SPEC")
    solve_opt = parse_block_content(raw_p1, "### SOLVE_OPTIMIZED")
    test_gen = parse_block_content(raw_p1, "### TEST_GENERATOR")

    if not spec or not solve_opt or not test_gen:
        return False, {}, model1, "", lat1, ret1, fb1, False, fb_reason1, ""

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


# ---------------------------------------------------------------------------
# V6/V7: BaseModelProber (session-scoped singleton)
# ---------------------------------------------------------------------------
_base_model_prober: Optional[BaseModelProber] = None


def _get_prober() -> BaseModelProber:
    global _base_model_prober
    if _base_model_prober is None:
        _base_model_prober = BaseModelProber()
    return _base_model_prober


def run_targeted_generation_pipeline(
    target_cluster: str,
    num_samples: int = 5,
    max_imbalance_ratio: float = 0.40,
    batch_size: int = 1,
    db_path: str = "data/v7_experiment_ledger.db",
    jsonl_path: str = "data/v7_approved_curriculum.jsonl",
    aux_jsonl_path: str = "data/v7_auxiliary_curriculum.jsonl",
    mode: str = "local",
    probe_manifest_path: str = "data/v7_probe_manifest.jsonl",
    metrics: Optional[PipelineMetrics] = None,
) -> None:
    """
    Failure-targeted curriculum generation loop with strict V7 primary eligibility gating.

    Args:
        target_cluster:      Failure cluster ID (e.g. ``"FC_01_MICRO_CORRECTNESS"``).
        num_samples:         Number of verified, deduplicated primary samples to generate.
        max_imbalance_ratio: Subtype imbalance ratio threshold for logging.
        batch_size:          Number of parallel generation calls per outer iteration.
        db_path:             SQLite ledger path.
        jsonl_path:          Primary approved curriculum JSONL path.
        aux_jsonl_path:      Auxiliary/non-primary curriculum JSONL path.
        mode:                Execution mode: 'local' (exports to probe manifest without loading Qwen)
                             or 'probe-direct' (requires Kaggle/GPU environment).
        probe_manifest_path: Path to write probe-required candidate manifest in local mode.
        metrics:             Optional PipelineMetrics instance.
    """
    from quality_gates import check_primary_targeted_eligibility

    print(f"=== STARTING TARGETED CURRICULUM GENERATION FOR CLUSTER: {target_cluster} (MODE: {mode.upper()}) ===")
    manifest = create_generation_manifest(output_path="data/generation_manifest.json")
    print(
        f"  [MANIFEST] Frozen manifest recorded (Freeze: {manifest.get('freeze_tag')}, "
        f"Primary Model: {manifest.get('generator_model')})"
    )

    oracle = VerificationOracle()
    dedup = StructuralDeduplicationEngine(similarity_threshold=0.70)
    ledger = ExperimentLedger(db_path=db_path, jsonl_path=jsonl_path, aux_jsonl_path=aux_jsonl_path)
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
            subtype = scheduler.select(ledger_dist=planned_dist, seed=b_seed, cluster_id=target_cluster)

            if total_in_cluster > 6:
                max_count = max(planned_dist.values()) if planned_dist else 0
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
        ) -> Tuple[int, str, bool, Dict[str, str], str, str, float, int, bool, bool, str, str]:
            """Worker thunk executed in the thread pool."""
            ok, blocks, m1, m2, latency, retries, fb1, fb2, fb_r1, fb_r2 = generate_targeted_candidate(
                cluster_id=cluster_id,
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
                    result = (b_seed, subtype, False, {}, "failed", "failed", 0.0, 0, False, False, "", "")
                    print(f"  [X] Batch worker exception (seed {b_seed}): {exc}")
                results_by_seed[b_seed] = result

        for b_seed in sorted(results_by_seed.keys()):
            if accepted >= num_samples:
                break

            b_seed_val, subtype, ok, blocks, m1, m2, latency, api_calls, fb1, fb2, fb_r1, fb_r2 = results_by_seed[b_seed]

            metrics.record_attempt(subtype=subtype, cluster=target_cluster)
            metrics.record_gen_latency(latency)
            metrics.record_initial_generation_api_calls(api_calls)
            metrics.record_model_serving(m1)

            gen_fallback_used = fb1 or fb2
            gen_fallback_reason = f"Pass1: {fb_r1} | Pass2: {fb_r2}".strip(" |") if gen_fallback_used else ""
            primary_model_name = EXPERIMENTAL_GENERATOR_POLICY["primary_model"]
            effective_model = primary_model_name if not gen_fallback_used else (f"{m1}+{m2}" if m1 != m2 else m1)

            if gen_fallback_used:
                print(
                    f"  [WARN] Fallback model used for seed {b_seed_val}: {gen_fallback_reason}",
                    flush=True,
                )

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
                        generator_model=effective_model,
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
                    generator_model=effective_model,
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
                    generator_model=effective_model,
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

            if mode == "local":
                # --- LOCAL MODE (Safe: Never downloads, loads or initializes Qwen) ---
                manifest_entry = {
                    "problem_id": problem_id,
                    "seed": b_seed_val,
                    "condition_type": "TARGETED",
                    "target_failure_cluster": target_cluster,
                    "monotonic_queue_subtype": subtype,
                    "spec": blocks["spec"],
                    "solve_opt": blocks["solve_opt"],
                    "solve_brute": blocks["solve_brute"],
                    "test_gen": blocks["test_gen"],
                    "prompt_hash": prompt_hash,
                    "generator_model": effective_model,
                    "generator_model_pass1": m1,
                    "generator_model_pass2": m2,
                    "generator_policy": "EXPERIMENTAL_GENERATOR_POLICY",
                    "fallback_used": gen_fallback_used,
                    "fallback_used_pass1": fb1,
                    "fallback_used_pass2": fb2,
                    "fallback_reason": gen_fallback_reason,
                    "fallback_reason_pass1": fb_r1,
                    "fallback_reason_pass2": fb_r2,
                    "verification_status": "APPROVED",
                    "phase_a_pass_count": ver_res["phase_a_pass_count"],
                    "phase_b_pass": ver_res["phase_b_pass"],
                    "oracle_confidence": ver_res["confidence"],
                    "ast_hash": dedup_res.ast_hash,
                    "cfg_hash": dedup_res.cfg_hash,
                    "similarity_score": dedup_res.similarity_score,
                    "protocol_version": "v7.0",
                    "probing_policy_version": "v1.0_qwen25_1.5b_instruct",
                    "probing_policy_hash": _PROBING_POLICY_HASH,
                    "base_model_id": "Qwen/Qwen2.5-1.5B-Instruct",
                }
                out_mfile = Path(probe_manifest_path)
                out_mfile.parent.mkdir(parents=True, exist_ok=True)
                with open(out_mfile, "a", encoding="utf-8") as mf:
                    mf.write(json.dumps(manifest_entry, sort_keys=True) + "\n")

                entry = LedgerEntry(
                    problem_id=problem_id,
                    seed=b_seed_val,
                    generator_model=effective_model,
                    generator_version="v1.0",
                    prompt_hash=prompt_hash,
                    target_failure_cluster=target_cluster,
                    monotonic_queue_subtype=subtype,
                    target_validation_status="PENDING_PROBE",
                    target_evidence="{}",
                    probe_result_hash="",
                    generator_policy="EXPERIMENTAL_GENERATOR_POLICY",
                    generator_model_pass1=m1,
                    generator_model_pass2=m2,
                    fallback_used=gen_fallback_used,
                    fallback_used_pass1=fb1,
                    fallback_used_pass2=fb2,
                    fallback_reason=gen_fallback_reason,
                    fallback_reason_pass1=fb_r1,
                    fallback_reason_pass2=fb_r2,
                    base_pass1=0.0,
                    base_pass10=0.0,
                    empirical_difficulty=0.0,
                    oracle_confidence=ver_res["confidence"],
                    phase_a_pass_count=ver_res["phase_a_pass_count"],
                    phase_b_pass=ver_res["phase_b_pass"],
                    verification_status="APPROVED",
                    ast_hash=dedup_res.ast_hash,
                    cfg_hash=dedup_res.cfg_hash,
                    similarity_score=dedup_res.similarity_score,
                    acceptance_reason="Verified locally; exported to probe manifest for Kaggle validation",
                    code_content=full_code,
                    spec=blocks["spec"],
                    primary_eligibility_status="AUXILIARY_NON_PRIMARY" if gen_fallback_used else "PENDING_KAGGLE_PROBE",
                    primary_eligibility_reason=(
                        "Generator fallback activated (fallback output cannot enter primary dataset)"
                        if gen_fallback_used
                        else "Verified locally; awaiting Kaggle base-model probe"
                    ),
                )
                ledger.record_entry(entry)

                if gen_fallback_used:
                    scheduler.record_outcome(subtype, approved=False)
                    print(
                        f"  [-] NON-PRIMARY / AUXILIARY (FALLBACK): {problem_id} "
                        f"— Reason: {entry.primary_eligibility_reason}"
                    )
                else:
                    accepted += 1
                    metrics.record_approved(subtype=subtype, cluster=target_cluster)
                    scheduler.record_outcome(subtype, approved=True)
                    print(
                        f"  [✓] EXPORTED FOR PROBING [{accepted}/{num_samples}]: {problem_id} "
                        f"-> {probe_manifest_path}"
                    )
                    print("      [LOCAL-GUARD] Qwen/Qwen2.5-1.5B-Instruct was NOT downloaded or initialized locally.")

            else:
                # --- DIRECT PROBE MODE (Executed on Kaggle/GPU server with override) ---
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

                import hashlib as _hl, json as _js
                probe_dict_json = _js.dumps(probe_result.to_dict(), sort_keys=True)
                probe_hash = _hl.sha256(probe_dict_json.encode()).hexdigest()[:16]

                target_validation_status: str
                if probe_result.failure_cluster_id == target_cluster:
                    target_validation_status = "VALIDATED"
                else:
                    target_validation_status = "UNVERIFIED"

                target_evidence_json = _js.dumps(probe_result.failure_evidence)

                entry = LedgerEntry(
                    problem_id=problem_id,
                    seed=b_seed_val,
                    generator_model=effective_model,
                    generator_version="v1.0",
                    prompt_hash=prompt_hash,
                    target_failure_cluster=target_cluster,
                    monotonic_queue_subtype=subtype,
                    target_validation_status=target_validation_status,
                    target_evidence=target_evidence_json,
                    probe_result_hash=probe_hash,
                    generator_policy="EXPERIMENTAL_GENERATOR_POLICY",
                    generator_model_pass1=m1,
                    generator_model_pass2=m2,
                    fallback_used=gen_fallback_used,
                    fallback_used_pass1=fb1,
                    fallback_used_pass2=fb2,
                    fallback_reason=gen_fallback_reason,
                    fallback_reason_pass1=fb_r1,
                    fallback_reason_pass2=fb_r2,
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
                    acceptance_reason=f"Approved candidate for {target_cluster} [target_validation={target_validation_status}]",
                    code_content=full_code,
                    spec=blocks["spec"],
                )

                eligibility = check_primary_targeted_eligibility(entry)
                entry.primary_eligibility_status = eligibility["status"]
                entry.primary_eligibility_reason = (
                    "; ".join(eligibility["reasons"]) if eligibility["reasons"] else "Meets all primary invariants"
                )

                ledger.record_entry(entry)

                if eligibility["is_primary"]:
                    accepted += 1
                    metrics.record_approved(subtype=subtype, cluster=target_cluster)
                    scheduler.record_outcome(subtype, approved=True)
                    print(
                        f"  [✓] ACCEPTED AS PRIMARY [{accepted}/{num_samples}]: {problem_id} "
                        f"(Diff: {diff:.2f}, AST: {dedup_res.ast_hash}, "
                        f"TargetValidation: {target_validation_status})"
                    )
                else:
                    scheduler.record_outcome(subtype, approved=False)
                    print(
                        f"  [-] NON-PRIMARY / AUXILIARY ({entry.primary_eligibility_status}): {problem_id} "
                        f"— Reason: {entry.primary_eligibility_reason}"
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
    parser.add_argument(
        "--mode",
        type=str,
        choices=["local", "probe-direct"],
        default="local",
        help="Execution mode: 'local' (default, exports probe-required manifest without downloading Qwen) or 'probe-direct'.",
    )
    parser.add_argument(
        "--probe-manifest",
        type=str,
        default="data/v7_probe_manifest.jsonl",
        help="Output JSONL manifest path for candidates awaiting Kaggle probing.",
    )
    args = parser.parse_args()

    run_targeted_generation_pipeline(
        target_cluster=args.target_failure_cluster,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        mode=args.mode,
        probe_manifest_path=args.probe_manifest,
    )

