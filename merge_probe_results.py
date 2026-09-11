"""merge_probe_results.py

Merges empirical Kaggle base-model probe results into the Phase 0 experiment ledger.

Enforces strict cryptographic provenance, verifies policy hashes, updates target
validation statuses, and performs final primary eligibility gating (Invariant I, III, IV, V).
Candidates lacking valid empirical base-model evidence are permanently kept out of the
primary curriculum.

Usage:
    python merge_probe_results.py \
        --probes data/v7_kaggle_probe_results.jsonl \
        --db data/v7_experiment_ledger.db \
        --approved-jsonl data/v7_approved_curriculum.jsonl \
        --aux-jsonl data/v7_auxiliary_curriculum.jsonl
"""

import argparse
import hashlib
import json
import os
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from base_model_prober import BASE_MODEL_ID, _PROBING_POLICY_HASH
from experiment_ledger import ExperimentLedger, LedgerEntry
from quality_gates import check_primary_eligibility


def merge_probe_results(
    probes_path: str = "data/v7_kaggle_probe_results.jsonl",
    db_path: str = "data/v7_experiment_ledger.db",
    approved_jsonl: str = "data/v7_approved_curriculum.jsonl",
    aux_jsonl: str = "data/v7_auxiliary_curriculum.jsonl",
) -> Dict[str, Any]:
    """
    Imports and merges Kaggle probe result records into the experiment ledger.
    """
    p_file = Path(probes_path)
    if not p_file.exists():
        raise FileNotFoundError(f"Probe results file not found: {probes_path}")

    ledger = ExperimentLedger(db_path=db_path, jsonl_path=approved_jsonl, aux_jsonl_path=aux_jsonl)

    # Read probe records
    probes: List[Dict[str, Any]] = []
    with open(p_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                probes.append(json.loads(line))

    print(f"=== MERGING KAGGLE PROBE RESULTS ===")
    print(f"  [INPUT] Probe results: {probes_path} ({len(probes)} records)")
    print(f"  [LEDGER] SQLite DB: {db_path}")
    print(f"  [TARGET JSONL] Primary Approved: {approved_jsonl}")
    print(f"  [AUX JSONL] Auxiliary: {aux_jsonl}")

    seen_problem_ids = set()
    primary_approved = 0
    auxiliary_routed = 0
    rejected_count = 0


    for idx, probe in enumerate(probes, 1):
        problem_id = probe.get("problem_id", "")
        if not problem_id:
            raise ValueError(f"Record {idx} missing problem_id")

        if problem_id in seen_problem_ids:
            raise ValueError(f"Duplicate probe result observed for problem_id: {problem_id}")
        seen_problem_ids.add(problem_id)

        # 1. Cryptographic Provenance & Policy Invariant Verification
        raw_probe = probe.get("raw_probe_result", {})
        recomputed_hash = hashlib.sha256(
            json.dumps(raw_probe, sort_keys=True).encode("utf-8")
        ).hexdigest()[:16]

        declared_hash = probe.get("probe_result_hash", "")
        if declared_hash != recomputed_hash:
            raise ValueError(
                f"Provenance hash mismatch for {problem_id}: "
                f"declared '{declared_hash}' != recomputed '{recomputed_hash}'"
            )

        model_id = probe.get("base_model_id", "")
        if model_id != BASE_MODEL_ID:
            raise ValueError(
                f"Base model mismatch for {problem_id}: '{model_id}' != '{BASE_MODEL_ID}'"
            )

        policy_hash = probe.get("probing_policy_hash", "")
        if policy_hash != _PROBING_POLICY_HASH:
            raise ValueError(
                f"Probing policy hash mismatch for {problem_id}: '{policy_hash}' != '{_PROBING_POLICY_HASH}'"
            )

        # 2. Retrieve Candidate Record from Ledger
        with sqlite3.connect(ledger.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM ledger WHERE problem_id = ?", (problem_id,))
            row = cursor.fetchone()

        if not row:
            raise KeyError(f"Candidate problem_id '{problem_id}' not found in ledger {db_path}")

        row_dict = dict(row)

        # 3. Update Empirical Probe Metrics
        target_val_status = probe.get("target_validation_status", "UNVERIFIED")
        target_evidence = probe.get("target_evidence", {})
        evidence_json = json.dumps(target_evidence) if isinstance(target_evidence, dict) else str(target_evidence)
        base_pass1 = float(probe.get("base_pass1", 0.0))
        base_pass10 = float(probe.get("base_pass10", 0.0))
        diff = float(probe.get("empirical_difficulty", 1.0 - base_pass10))

        row_dict["base_pass1"] = base_pass1
        row_dict["base_pass10"] = base_pass10
        row_dict["empirical_difficulty"] = diff
        row_dict["target_validation_status"] = target_val_status
        row_dict["target_evidence"] = evidence_json
        row_dict["probe_result_hash"] = declared_hash

        # 4. Authoritative Primary Eligibility Gate
        cond_type = probe.get("condition_type", "TARGETED")
        eligibility = check_primary_eligibility(row_dict, condition_type=cond_type)
        status = eligibility["status"]
        reasons = "; ".join(eligibility["reasons"]) if eligibility["reasons"] else "Meets all primary invariants"

        row_dict["primary_eligibility_status"] = status
        row_dict["primary_eligibility_reason"] = reasons

        # 5. Persist Update to SQLite Ledger
        with sqlite3.connect(ledger.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE ledger SET
                    base_pass1 = ?,
                    base_pass10 = ?,
                    empirical_difficulty = ?,
                    target_validation_status = ?,
                    target_evidence = ?,
                    probe_result_hash = ?,
                    primary_eligibility_status = ?,
                    primary_eligibility_reason = ?
                WHERE problem_id = ?
                """,
                (
                    base_pass1,
                    base_pass10,
                    diff,
                    target_val_status,
                    evidence_json,
                    declared_hash,
                    status,
                    reasons,
                    problem_id,
                ),
            )
            conn.commit()

        # 6. Route to JSONL Dataset Namespaces
        entry = LedgerEntry(
            problem_id=row_dict["problem_id"],
            seed=int(row_dict["seed"]),
            generator_model=row_dict["generator_model"],
            generator_version=row_dict.get("generator_version", "v1.0"),
            prompt_hash=row_dict["prompt_hash"],
            target_failure_cluster=row_dict["target_failure_cluster"],
            monotonic_queue_subtype=row_dict["monotonic_queue_subtype"],
            target_validation_status=target_val_status,
            target_evidence=evidence_json,
            probe_result_hash=declared_hash,
            generator_policy=row_dict.get("generator_policy", "EXPERIMENTAL_GENERATOR_POLICY"),
            generator_model_pass1=row_dict.get("generator_model_pass1", row_dict["generator_model"]),
            generator_model_pass2=row_dict.get("generator_model_pass2", row_dict["generator_model"]),
            fallback_used=bool(row_dict.get("fallback_used", False)),
            fallback_used_pass1=bool(row_dict.get("fallback_used_pass1", False)),
            fallback_used_pass2=bool(row_dict.get("fallback_used_pass2", False)),
            fallback_reason=row_dict.get("fallback_reason", ""),
            fallback_reason_pass1=row_dict.get("fallback_reason_pass1", ""),
            fallback_reason_pass2=row_dict.get("fallback_reason_pass2", ""),
            base_pass1=base_pass1,
            base_pass10=base_pass10,
            empirical_difficulty=diff,
            oracle_confidence=float(row_dict.get("oracle_confidence", 1.0)),
            phase_a_pass_count=int(row_dict.get("phase_a_pass_count", 0)),
            phase_b_pass=bool(row_dict.get("phase_b_pass", True)),
            verification_status=row_dict.get("verification_status", "APPROVED"),
            ast_hash=row_dict.get("ast_hash", ""),
            cfg_hash=row_dict.get("cfg_hash", ""),
            similarity_score=float(row_dict.get("similarity_score", 0.0)),
            acceptance_reason=row_dict.get("acceptance_reason", ""),
            code_content=row_dict.get("code_content", ""),
            spec=row_dict.get("spec", ""),
            primary_eligibility_status=status,
            primary_eligibility_reason=reasons,
        )

        if eligibility["is_primary"]:
            primary_approved += 1
            out_path = approved_jsonl
            print(
                f"  [✓] MERGED AS PRIMARY: {problem_id} "
                f"(Diff: {diff:.2f}, Target: {target_val_status}) -> {approved_jsonl}"
            )
        else:
            auxiliary_routed += 1
            out_path = aux_jsonl
            print(
                f"  [-] MERGED AS AUXILIARY ({status}): {problem_id} "
                f"— {reasons} -> {aux_jsonl}"
            )

        # Write to JSONL
        target_f = Path(out_path)
        target_f.parent.mkdir(parents=True, exist_ok=True)
        with open(target_f, "a", encoding="utf-8") as jf:
            jf.write(json.dumps(entry.to_dict(), sort_keys=True) + "\n")

    summary = {
        "total_merged": len(probes),
        "primary_approved": primary_approved,
        "auxiliary_routed": auxiliary_routed,
        "db_path": db_path,
        "approved_jsonl": approved_jsonl,
        "aux_jsonl": aux_jsonl,
    }

    print(f"\n[✓] MERGE COMPLETE: {len(probes)} records processed.")
    print(f"    Primary Approved: {primary_approved}")
    print(f"    Auxiliary / Non-Primary: {auxiliary_routed}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge Kaggle Probe Results into Phase 0 Ledger")
    parser.add_argument(
        "--probes",
        type=str,
        default="data/v7_kaggle_probe_results.jsonl",
        help="Path to Kaggle probe results JSONL file.",
    )
    parser.add_argument(
        "--db",
        type=str,
        default="data/v7_experiment_ledger.db",
        help="Path to SQLite experiment ledger DB.",
    )
    parser.add_argument(
        "--approved-jsonl",
        type=str,
        default="data/v7_approved_curriculum.jsonl",
        help="Path to primary approved curriculum JSONL file.",
    )
    parser.add_argument(
        "--aux-jsonl",
        type=str,
        default="data/v7_auxiliary_curriculum.jsonl",
        help="Path to auxiliary/non-primary curriculum JSONL file.",
    )
    args = parser.parse_args()

    merge_probe_results(
        probes_path=args.probes,
        db_path=args.db,
        approved_jsonl=args.approved_jsonl,
        aux_jsonl=args.aux_jsonl,
    )
