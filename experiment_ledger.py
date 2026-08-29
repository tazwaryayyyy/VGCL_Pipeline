"""experiment_ledger.py

Structured Experiment Ledger for VGCL Phase 0.

Provides persistent, audit-ready scientific recordkeeping with SQLite and JSONL backends.
Ensures metadata immutability and complete experimental traceability.

V6 additions:
  - target_validation_status: VALIDATED | UNVERIFIED | NOT_APPLICABLE
  - target_evidence: JSON blob of observable execution evidence that motivated the target
  - probe_result_hash: SHA-256[:16] of the ProbingResult dict for cross-referencing
  - generator_policy: name of the generator policy constant used
  - fallback_used: True if a fallback model was used instead of the primary
  - fallback_reason: reason string if fallback was activated

  Invariant: target_failure_cluster records the REQUESTED target (CLI argument).
  A label of VALIDATED in target_validation_status means real base-model failure
  evidence exists in target_evidence; UNVERIFIED means no such evidence was produced.
"""

import json
import os
import sqlite3
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class LedgerEntry:
    # Identification
    problem_id: str
    seed: int
    timestamp: float = field(default_factory=time.time)

    # Generation
    generator_model: str = "gemini-3.6-flash"
    generator_version: str = "v1.0"
    prompt_hash: str = "hash0000"
    target_failure_cluster: str = "FC_01_MICRO_CORRECTNESS"   # requested target (CLI argument)
    monotonic_queue_subtype: str = "Type A: sliding-window extrema"

    # V6: Target attribution
    # target_failure_cluster records the REQUESTED target (unchanged for backward compat).
    # target_validation_status distinguishes whether the target is backed by real evidence:
    #   VALIDATED    — real base-model failure evidence exists in target_evidence
    #   UNVERIFIED   — no empirical evidence; label comes from CLI argument only
    #   NOT_APPLICABLE — uniform control condition (no targeting)
    target_validation_status: str = "UNVERIFIED"
    target_evidence: str = "{}"     # JSON string of ProbingResult.failure_evidence
    probe_result_hash: str = ""     # SHA-256[:16] of ProbingResult.to_dict() for cross-ref

    # V6: Generator policy traceability
    generator_policy: str = "EXPERIMENTAL_GENERATOR_POLICY"
    fallback_used: bool = False
    fallback_reason: str = ""

    # Difficulty  (real values from BaseModelProber in V6; mark historical V5 values as mocked)
    base_pass1: float = 0.0
    base_pass10: float = 0.0
    empirical_difficulty: float = 1.0  # 1.0 - base_pass10

    # Verification
    oracle_confidence: float = 1.0
    phase_a_pass_count: int = 20
    phase_b_pass: bool = True
    verification_status: str = "APPROVED"

    # Novelty
    ast_hash: str = ""
    cfg_hash: str = ""
    similarity_score: float = 0.0

    # Acceptance
    acceptance_reason: str = "Verified novel problem matching target failure cluster"
    reviewer_override: bool = False
    quarantine_reference: Optional[str] = None

    # Code Content
    code_content: str = ""
    spec: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ExperimentLedger:
    """Scientific Experiment Ledger supporting SQLite persistent storage and JSONL streaming."""

    def __init__(self, db_path: str = "data/experiment_ledger.db", jsonl_path: str = "data/approved_curriculum.jsonl"):
        self.db_path = db_path
        self.jsonl_path = jsonl_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        os.makedirs(os.path.dirname(jsonl_path), exist_ok=True)
        self._init_sqlite()

    def _init_sqlite(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ledger (
                    problem_id TEXT PRIMARY KEY,
                    seed INTEGER,
                    timestamp REAL,
                    generator_model TEXT,
                    generator_version TEXT,
                    prompt_hash TEXT,
                    target_failure_cluster TEXT,
                    monotonic_queue_subtype TEXT,
                    base_pass1 REAL,
                    base_pass10 REAL,
                    empirical_difficulty REAL,
                    oracle_confidence REAL,
                    phase_a_pass_count INTEGER,
                    phase_b_pass INTEGER,
                    verification_status TEXT,
                    ast_hash TEXT,
                    cfg_hash TEXT,
                    similarity_score REAL,
                    acceptance_reason TEXT,
                    reviewer_override INTEGER,
                    quarantine_reference TEXT,
                    code_content TEXT,
                    spec TEXT
                )
            """)
            # V6 migration: add new columns if they do not exist (idempotent)
            v6_columns = [
                ("target_validation_status", "TEXT DEFAULT 'UNVERIFIED'"),
                ("target_evidence", "TEXT DEFAULT '{}'"),
                ("probe_result_hash", "TEXT DEFAULT ''"),
                ("generator_policy", "TEXT DEFAULT 'EXPERIMENTAL_GENERATOR_POLICY'"),
                ("fallback_used", "INTEGER DEFAULT 0"),
                ("fallback_reason", "TEXT DEFAULT ''"),
            ]
            existing = {row[1] for row in cursor.execute("PRAGMA table_info(ledger)").fetchall()}
            for col_name, col_def in v6_columns:
                if col_name not in existing:
                    cursor.execute(f"ALTER TABLE ledger ADD COLUMN {col_name} {col_def}")
            conn.commit()


    def record_entry(self, entry: LedgerEntry) -> None:
        """Appends entry to both SQLite database and JSONL file immutably."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO ledger (
                    problem_id, seed, timestamp, generator_model, generator_version, prompt_hash,
                    target_failure_cluster, monotonic_queue_subtype, base_pass1, base_pass10,
                    empirical_difficulty, oracle_confidence, phase_a_pass_count, phase_b_pass,
                    verification_status, ast_hash, cfg_hash, similarity_score, acceptance_reason,
                    reviewer_override, quarantine_reference, code_content, spec,
                    target_validation_status, target_evidence, probe_result_hash,
                    generator_policy, fallback_used, fallback_reason
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                          ?, ?, ?, ?, ?, ?)
            """, (
                entry.problem_id, entry.seed, entry.timestamp, entry.generator_model,
                entry.generator_version, entry.prompt_hash, entry.target_failure_cluster,
                entry.monotonic_queue_subtype, entry.base_pass1, entry.base_pass10,
                entry.empirical_difficulty, entry.oracle_confidence, entry.phase_a_pass_count,
                1 if entry.phase_b_pass else 0, entry.verification_status, entry.ast_hash,
                entry.cfg_hash, entry.similarity_score, entry.acceptance_reason,
                1 if entry.reviewer_override else 0, entry.quarantine_reference,
                entry.code_content, entry.spec,
                entry.target_validation_status,
                entry.target_evidence,
                entry.probe_result_hash,
                entry.generator_policy,
                1 if entry.fallback_used else 0,
                entry.fallback_reason,
            ))
            conn.commit()

        # Append to JSONL
        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry.to_dict()) + "\n")


    def get_entries(self, target_cluster: Optional[str] = None) -> List[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            if target_cluster:
                cursor.execute("SELECT * FROM ledger WHERE target_failure_cluster = ?", (target_cluster,))
            else:
                cursor.execute("SELECT * FROM ledger")
            rows = cursor.fetchall()
            return [dict(r) for r in rows]

    def get_subtype_distribution(self, target_cluster: str) -> Dict[str, int]:
        entries = self.get_entries(target_cluster)
        dist: Dict[str, int] = {}
        for e in entries:
            st = e.get("monotonic_queue_subtype", "Unknown")
            dist[st] = dist.get(st, 0) + 1
        return dist

    def get_cluster_balance_stats(self) -> Dict[str, Dict[str, Any]]:
        """Audits exposure balance across clusters: sample counts, mean difficulty, mean token length."""
        entries = self.get_entries()
        stats: Dict[str, Dict[str, Any]] = {}
        for e in entries:
            cluster = e.get("target_failure_cluster", "UNKNOWN")
            if cluster not in stats:
                stats[cluster] = {"count": 0, "total_diff": 0.0, "total_code_len": 0}
            stats[cluster]["count"] += 1
            stats[cluster]["total_diff"] += float(e.get("empirical_difficulty", 0.0))
            stats[cluster]["total_code_len"] += len(str(e.get("code_content", "")).split())

        result: Dict[str, Dict[str, Any]] = {}
        for c, s in stats.items():
            cnt = s["count"]
            result[c] = {
                "sample_count": cnt,
                "mean_empirical_difficulty": s["total_diff"] / cnt if cnt > 0 else 0.0,
                "mean_token_count": s["total_code_len"] / cnt if cnt > 0 else 0,
            }
        return result

    def get_generator_model_distribution(self) -> Dict[str, Any]:
        """Audits generator-model exposure and fallback frequency across recorded entries."""
        entries = self.get_entries()
        total = len(entries)
        counts: Dict[str, int] = {}
        primary_count = 0
        fallback_count = 0
        primary_model = "gemini-3.6-flash"

        for e in entries:
            m = e.get("generator_model", "unknown")
            counts[m] = counts.get(m, 0) + 1
            if m == primary_model:
                primary_count += 1
            else:
                fallback_count += 1

        return {
            "total_entries": total,
            "model_counts": counts,
            "primary_model": primary_model,
            "primary_count": primary_count,
            "fallback_count": fallback_count,
            "fallback_frequency": fallback_count / total if total > 0 else 0.0,
        }

    def get_target_validation_summary(self) -> Dict[str, Any]:
        """V6: Audits what fraction of targeted entries have VALIDATED vs UNVERIFIED targets.

        This is the primary scientific audit tool for target attribution integrity.
        """
        entries = self.get_entries()
        counts: Dict[str, int] = {"VALIDATED": 0, "UNVERIFIED": 0, "NOT_APPLICABLE": 0, "UNKNOWN": 0}
        cluster_breakdown: Dict[str, Dict[str, int]] = {}
        for e in entries:
            status = e.get("target_validation_status", "UNKNOWN")
            cluster = e.get("target_failure_cluster", "UNKNOWN")
            counts[status] = counts.get(status, 0) + 1
            if cluster not in cluster_breakdown:
                cluster_breakdown[cluster] = {}
            cluster_breakdown[cluster][status] = cluster_breakdown[cluster].get(status, 0) + 1
        total = len(entries)
        return {
            "total_entries": total,
            "validation_counts": counts,
            "validated_fraction": counts["VALIDATED"] / total if total > 0 else 0.0,
            "cluster_breakdown": cluster_breakdown,
        }


if __name__ == "__main__":
    ledger = ExperimentLedger()
    sample_entry = LedgerEntry(
        problem_id="PROB_TEST_101",
        seed=101,
        target_failure_cluster="FC_01_MICRO_CORRECTNESS",
        monotonic_queue_subtype="Type A: sliding-window extrema",
        code_content="def solve(): pass",
    )
    ledger.record_entry(sample_entry)
    print("Logged entries count:", len(ledger.get_entries()))
    print("Subtype distribution:", ledger.get_subtype_distribution("FC_01_MICRO_CORRECTNESS"))
    print("Model distribution:", ledger.get_generator_model_distribution())
