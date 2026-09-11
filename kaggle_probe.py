"""kaggle_probe.py

Kaggle-side Probing Script for VGCL Phase 0.

Executes real base-model probing (Qwen/Qwen2.5-1.5B-Instruct) in the Kaggle GPU
environment on candidates generated locally. Preserves strict causal attribution,
frozen probing policy parameters, deterministic failure evidence classification,
and cryptographic provenance hashes without running or downloading model weights locally.

Usage:
    python kaggle_probe.py --input data/v7_probe_manifest.jsonl --output data/v7_kaggle_probe_results.jsonl
"""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

from base_model_prober import (
    BASE_MODEL_ID,
    PROBING_POLICY,
    _PROBING_POLICY_HASH,
    BaseModelProber,
    ProbingResult,
)


def run_kaggle_probing(
    manifest_path: str = "data/v7_probe_manifest.jsonl",
    output_path: str = "data/v7_kaggle_probe_results.jsonl",
    device: Optional[str] = None,
    limit: Optional[int] = None,
    prober: Optional[BaseModelProber] = None,
) -> Dict[str, Any]:
    """
    Reads candidates from manifest_path, executes real base-model probing on GPU,
    and writes deterministic probe result records to output_path.
    """
    in_file = Path(manifest_path)
    if not in_file.exists():
        raise FileNotFoundError(f"Probe manifest not found: {manifest_path}")

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # Read candidates
    candidates = []
    with open(in_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                candidates.append(json.loads(line))

    if limit is not None:
        candidates = candidates[:limit]

    print(f"=== STARTING KAGGLE BASE-MODEL PROBING ===")
    print(f"  [INPUT] Manifest: {manifest_path} ({len(candidates)} candidates)")
    print(f"  [OUTPUT] Destination: {output_path}")
    print(f"  [POLICY] Model: {BASE_MODEL_ID} | Hash: {_PROBING_POLICY_HASH} | k={PROBING_POLICY['k_samples']}")

    # Instantiate prober (authorized for Kaggle / probe environment)
    if prober is None:
        prober = BaseModelProber(device=device, allow_local_probe=True)

    results = []
    validated_count = 0
    unverified_count = 0

    t0 = time.perf_counter()
    for idx, c in enumerate(candidates, 1):
        problem_id = c["problem_id"]
        cond = c.get("condition_type", "TARGETED")
        target_cluster = c["target_failure_cluster"]
        spec = c["spec"]
        solve_brute = c["solve_brute"]
        test_gen = c["test_gen"]
        seed = int(c["seed"])

        print(f"[{idx}/{len(candidates)}] Probing {problem_id} ({cond} - {target_cluster}) ...", flush=True)

        probe_res: ProbingResult = prober.probe(
            problem_id=problem_id,
            problem_spec=spec,
            solve_brute=solve_brute,
            test_gen=test_gen,
            seed=seed,
        )

        probe_dict = probe_res.to_dict()
        probe_result_hash = hashlib.sha256(
            json.dumps(probe_dict, sort_keys=True).encode("utf-8")
        ).hexdigest()[:16]

        if cond == "TARGETED":
            if probe_res.failure_cluster_id == target_cluster:
                target_validation_status = "VALIDATED"
                validated_count += 1
            else:
                target_validation_status = "UNVERIFIED"
                unverified_count += 1
        else:
            target_validation_status = "NOT_APPLICABLE"

        record: Dict[str, Any] = {
            "problem_id": problem_id,
            "condition_type": cond,
            "target_failure_cluster": target_cluster,
            "monotonic_queue_subtype": c.get("monotonic_queue_subtype", ""),
            "target_validation_status": target_validation_status,
            "target_evidence": probe_res.failure_evidence,
            "base_pass1": probe_res.pass1,
            "base_pass10": probe_res.pass10,
            "empirical_difficulty": probe_res.empirical_difficulty,
            "probe_result_hash": probe_result_hash,
            "probe_timestamp": probe_res.probe_timestamp,
            "base_model_id": probe_res.base_model_id,
            "probing_policy_version": probe_res.probing_policy_version,
            "probing_policy_hash": probe_res.probing_policy_hash,
            "raw_probe_result": probe_dict,
        }

        with open(out_file, "a", encoding="utf-8") as out_f:
            out_f.write(json.dumps(record, sort_keys=True) + "\n")

        results.append(record)
        print(
            f"      Diff={probe_res.empirical_difficulty:.2f} | "
            f"Pass@1={probe_res.pass1:.0f} | Pass@10={probe_res.pass10:.0f} | "
            f"Status={target_validation_status} (Observed={probe_res.failure_cluster_id})",
            flush=True,
        )

    elapsed = time.perf_counter() - t0
    summary = {
        "total_probed": len(candidates),
        "validated_count": validated_count,
        "unverified_count": unverified_count,
        "elapsed_seconds": elapsed,
        "output_path": output_path,
    }
    print(f"\n[✓] KAGGLE PROBING COMPLETE in {elapsed:.1f}s")
    print(f"    Total: {len(candidates)} | Validated: {validated_count} | Unverified: {unverified_count}")
    print(f"    Results saved to: {output_path}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kaggle Base-Model Prober for VGCL Phase 0")
    parser.add_argument(
        "--input",
        type=str,
        default="data/v7_probe_manifest.jsonl",
        help="Path to input candidate manifest generated locally.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/v7_kaggle_probe_results.jsonl",
        help="Path to write auditable probe result records.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Execution device ('cuda' or 'cpu'). Defaults to auto-detect.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max candidates to probe.",
    )
    args = parser.parse_args()

    run_kaggle_probing(
        manifest_path=args.input,
        output_path=args.output,
        device=args.device,
        limit=args.limit,
    )
