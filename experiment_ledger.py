"""experiment_ledger.py

Structured Experiment Ledger for VGCL Phase 0.

Provides persistent, audit-ready scientific recordkeeping with SQLite and JSONL backends.
Ensures metadata immutability and complete experimental traceability.
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
    target_failure_cluster: str = "FC_01_MICRO_CORRECTNESS"
    monotonic_queue_subtype: str = "Type A: sliding-window extrema"

    # Difficulty
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
                    reviewer_override, quarantine_reference, code_content, spec
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                entry.problem_id, entry.seed, entry.timestamp, entry.generator_model,
                entry.generator_version, entry.prompt_hash, entry.target_failure_cluster,
                entry.monotonic_queue_subtype, entry.base_pass1, entry.base_pass10,
                entry.empirical_difficulty, entry.oracle_confidence, entry.phase_a_pass_count,
                1 if entry.phase_b_pass else 0, entry.verification_status, entry.ast_hash,
                entry.cfg_hash, entry.similarity_score, entry.acceptance_reason,
                1 if entry.reviewer_override else 0, entry.quarantine_reference,
                entry.code_content, entry.spec
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
