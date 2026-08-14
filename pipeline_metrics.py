"""pipeline_metrics.py

Thread-safe, append-only performance metrics collector for VGCL Phase 0.

Tracks generation latency, verification latency, approval rates, syntax pass
rates, phase pass rates, deduplication rejection rates, and API call counts.
Writes structured JSON and optional CSV summaries without affecting pipeline
experimental semantics.

All counter mutations are protected by a threading.Lock so the module is safe
for use with concurrent.futures.ThreadPoolExecutor batch generation.
"""

import csv
import json
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional


@dataclass
class _Counters:
    """Mutable atomic counters for pipeline events."""

    attempts: int = 0               # Total generation attempts
    gen_errors: int = 0             # Generation returned empty/invalid blocks
    syntax_failures: int = 0        # Pre-oracle AST parse failures (after repair)
    syntax_repairs: int = 0         # Successful single-attempt repairs
    oracle_attempts: int = 0        # Samples submitted to oracle
    phase_a_passes: int = 0         # Oracle Phase A passed
    phase_b_passes: int = 0         # Oracle Phase B passed (= verified)
    dedup_rejections: int = 0       # Structural deduplication rejections
    approved: int = 0               # Samples accepted into ledger
    api_calls: int = 0              # Total LLM API calls made
    initial_generation_api_calls: int = 0
    syntax_retry_api_calls: int = 0
    syntax_retry_successes: int = 0
    reference_independence_rejections: int = 0
    subtype_rejections: int = 0
    phase_a_failures: int = 0
    phase_b_failures: int = 0

    # Per-subtype approved / attempted counters
    subtype_approved: Dict[str, int] = field(default_factory=dict)
    subtype_attempted: Dict[str, int] = field(default_factory=dict)

    # Per-cluster approved / attempted counters
    cluster_approved: Dict[str, int] = field(default_factory=dict)
    cluster_attempted: Dict[str, int] = field(default_factory=dict)


class PipelineMetrics:
    """
    Thread-safe metrics aggregator for one pipeline run.

    Usage::

        metrics = PipelineMetrics()
        metrics.record_attempt(subtype="Type A: ...", cluster="FC_01_...")
        metrics.record_gen_latency(3.2)
        metrics.record_api_calls(2)
        metrics.record_oracle_attempt()
        metrics.record_phase_a_pass()
        metrics.record_phase_b_pass()
        metrics.record_approved(subtype="Type A: ...", cluster="FC_01_...")
        metrics.flush("data/metrics_run_001.json")
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters = _Counters()
        self._gen_latencies: list = []
        self._ver_latencies: list = []
        self._start_time: float = time.perf_counter()

    # ------------------------------------------------------------------
    # Counter mutation helpers (all thread-safe)
    # ------------------------------------------------------------------

    def record_attempt(self, subtype: str = "", cluster: str = "") -> None:
        """Records one generation attempt, keyed by subtype and cluster."""
        with self._lock:
            self._counters.attempts += 1
            if subtype:
                self._counters.subtype_attempted[subtype] = (
                    self._counters.subtype_attempted.get(subtype, 0) + 1
                )
            if cluster:
                self._counters.cluster_attempted[cluster] = (
                    self._counters.cluster_attempted.get(cluster, 0) + 1
                )

    def record_gen_error(self) -> None:
        """Records a generation-side block parsing failure."""
        with self._lock:
            self._counters.gen_errors += 1

    def record_gen_latency(self, latency_s: float) -> None:
        """Records wall-clock generation latency for one sample (both passes)."""
        with self._lock:
            self._gen_latencies.append(latency_s)

    def record_api_calls(self, count: int) -> None:
        """Accumulates LLM API call count for one sample."""
        with self._lock:
            self._counters.api_calls += count

    def record_initial_generation_api_calls(self, count: int) -> None:
        """Records API calls used by initial two-pass generation."""
        with self._lock:
            self._counters.initial_generation_api_calls += count
            self._counters.api_calls += count

    def record_syntax_retry_api_calls(self, count: int) -> None:
        """Records API calls used by the one allowed syntax retry."""
        with self._lock:
            self._counters.syntax_retry_api_calls += count
            self._counters.api_calls += count

    def record_syntax_retry_success(self) -> None:
        """Records a syntax retry that produced parseable package code."""
        with self._lock:
            self._counters.syntax_retry_successes += 1

    def record_syntax_failure(self) -> None:
        """Records one sample that failed pre-oracle syntax validation."""
        with self._lock:
            self._counters.syntax_failures += 1

    def record_syntax_repair(self) -> None:
        """Records one sample that required (and passed) a single repair attempt."""
        with self._lock:
            self._counters.syntax_repairs += 1

    def record_oracle_attempt(self) -> None:
        """Records submission of one sample to the verification oracle."""
        with self._lock:
            self._counters.oracle_attempts += 1

    def record_verification_latency(self, latency_s: float) -> None:
        """Records wall-clock oracle verification latency for one sample."""
        with self._lock:
            self._ver_latencies.append(latency_s)

    def record_phase_a_pass(self) -> None:
        """Records one Phase A pass."""
        with self._lock:
            self._counters.phase_a_passes += 1

    def record_phase_a_failure(self) -> None:
        """Records one Phase A oracle failure."""
        with self._lock:
            self._counters.phase_a_failures += 1

    def record_phase_b_pass(self) -> None:
        """Records one Phase B pass (i.e., fully verified)."""
        with self._lock:
            self._counters.phase_b_passes += 1

    def record_phase_b_failure(self) -> None:
        """Records one Phase B oracle failure."""
        with self._lock:
            self._counters.phase_b_failures += 1

    def record_dedup_rejection(self) -> None:
        """Records one deduplication rejection."""
        with self._lock:
            self._counters.dedup_rejections += 1

    def record_reference_independence_rejection(self) -> None:
        """Records optimized/reference structural-independence rejection."""
        with self._lock:
            self._counters.reference_independence_rejections += 1

    def record_subtype_rejection(self) -> None:
        """Records deterministic subtype-validation rejection."""
        with self._lock:
            self._counters.subtype_rejections += 1

    def record_approved(self, subtype: str = "", cluster: str = "") -> None:
        """Records one sample accepted into the ledger."""
        with self._lock:
            self._counters.approved += 1
            if subtype:
                self._counters.subtype_approved[subtype] = (
                    self._counters.subtype_approved.get(subtype, 0) + 1
                )
            if cluster:
                self._counters.cluster_approved[cluster] = (
                    self._counters.cluster_approved.get(cluster, 0) + 1
                )

    # ------------------------------------------------------------------
    # Summary computation (produces a plain dict snapshot)
    # ------------------------------------------------------------------

    def _safe_mean(self, values: list) -> float:
        return sum(values) / len(values) if values else 0.0

    def snapshot(self) -> Dict[str, Any]:
        """
        Returns a thread-safe point-in-time metrics snapshot as a plain dict.

        All rates are floats in [0.0, 1.0]. Latencies are in seconds.
        """
        with self._lock:
            c = self._counters
            attempts = c.attempts or 1  # guard zero-division

            # Approval rate: approved / attempts
            approval_rate = c.approved / attempts

            # Syntax is attempted only after generation produced parseable blocks.
            syntax_attempts = c.attempts - c.gen_errors
            syntax_passes = syntax_attempts - c.syntax_failures
            syntax_denominator = syntax_attempts or 1
            syntax_pass_rate = syntax_passes / syntax_denominator

            # Phase A pass rate: phase_a_passes / oracle_attempts
            oracle_attempts = c.oracle_attempts or 1
            phase_a_rate = c.phase_a_passes / oracle_attempts

            # Phase B pass rate: phase_b_passes / phase_a_passes
            phase_a = c.phase_a_passes or 1
            phase_b_rate = c.phase_b_passes / phase_a

            # Dedup is attempted only after Phase B verification succeeds.
            dedup_submissions = c.phase_b_passes
            dedup_denominator = dedup_submissions or 1
            dedup_rate = c.dedup_rejections / dedup_denominator

            # API calls per approved
            approved = c.approved or 1
            api_per_approved = c.api_calls / approved

            # Per-subtype approval rates
            subtype_rates: Dict[str, float] = {}
            for st, att in c.subtype_attempted.items():
                apr = c.subtype_approved.get(st, 0)
                subtype_rates[st] = apr / att if att > 0 else 0.0

            # Per-cluster approval rates
            cluster_rates: Dict[str, float] = {}
            for cl, att in c.cluster_attempted.items():
                apr = c.cluster_approved.get(cl, 0)
                cluster_rates[cl] = apr / att if att > 0 else 0.0

            gen_lats = list(self._gen_latencies)
            ver_lats = list(self._ver_latencies)

        elapsed = time.perf_counter() - self._start_time

        return {
            "wall_clock_elapsed_s": round(elapsed, 3),
            "attempts": c.attempts,
            "approved": c.approved,
            "gen_errors": c.gen_errors,
            "syntax_attempts": syntax_attempts,
            "syntax_passes": syntax_passes,
            "syntax_failures": c.syntax_failures,
            "syntax_repairs": c.syntax_repairs,
            "oracle_attempts": c.oracle_attempts,
            "phase_a_passes": c.phase_a_passes,
            "phase_b_submissions": c.phase_a_passes,
            "phase_b_passes": c.phase_b_passes,
            "dedup_submissions": dedup_submissions,
            "dedup_rejections": c.dedup_rejections,
            "api_calls": c.api_calls,
            "generation_attempts": c.attempts,
            "initial_generation_api_calls": c.initial_generation_api_calls,
            "syntax_retry_api_calls": c.syntax_retry_api_calls,
            "total_generation_api_calls": c.api_calls,
            "syntax_retry_successes": c.syntax_retry_successes,
            "reference_independence_rejections": c.reference_independence_rejections,
            "subtype_rejections": c.subtype_rejections,
            "phase_a_failures": c.phase_a_failures,
            "phase_b_failures": c.phase_b_failures,
            # Rates
            "approval_rate": round(approval_rate, 6),
            "syntax_pass_rate": round(syntax_pass_rate, 6),
            "phase_a_pass_rate": round(phase_a_rate, 6),
            "phase_b_pass_rate": round(phase_b_rate, 6),
            "dedup_rejection_rate": round(dedup_rate, 6),
            "api_calls_per_approved": round(api_per_approved, 3),
            # Latencies
            "mean_gen_latency_s": round(self._safe_mean(gen_lats), 3),
            "mean_verification_latency_s": round(self._safe_mean(ver_lats), 3),
            # Breakdown
            "approval_rate_by_subtype": {
                k: round(v, 6) for k, v in subtype_rates.items()
            },
            "approval_rate_by_cluster": {
                k: round(v, 6) for k, v in cluster_rates.items()
            },
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def flush(self, json_path: str, csv_path: Optional[str] = None) -> None:
        """
        Writes the current metrics snapshot to a JSON file and optional CSV.

        The flush is best-effort: a write failure is printed but does not
        raise, so a disk-full condition cannot abort the pipeline.

        Args:
            json_path: Absolute or relative path for the JSON output file.
            csv_path:  Optional path for CSV output. If None, CSV is skipped.
        """
        snap = self.snapshot()
        try:
            import os
            os.makedirs(os.path.dirname(json_path) if os.path.dirname(json_path) else ".", exist_ok=True)
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(snap, f, indent=2)
        except Exception as exc:
            print(f"  [METRICS] Warning: could not write JSON metrics to '{json_path}': {exc}")

        if csv_path is None:
            return

        # CSV: flat scalar rows only (skip nested dicts)
        scalar_rows = {k: v for k, v in snap.items() if not isinstance(v, dict)}
        try:
            import os
            os.makedirs(os.path.dirname(csv_path) if os.path.dirname(csv_path) else ".", exist_ok=True)
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(scalar_rows.keys()))
                writer.writeheader()
                writer.writerow(scalar_rows)
        except Exception as exc:
            print(f"  [METRICS] Warning: could not write CSV metrics to '{csv_path}': {exc}")
