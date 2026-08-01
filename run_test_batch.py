import io
import json
import os
import sys
import time
from typing import Dict, List

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from dedup import CorpusBalanceEngine
from runner import execute_retry_loop

PARADIGMS = [
    "Dynamic Programming & Monotonic Queue",
    "Graph Traversals & DFS/BFS",
    "Disjoint Set Union (DSU) & Greedy Graph Analysis",
    "Sliding Window & Two Pointers",
    "Matrix Grid Traversals & Dynamic Programming",
    "Min-Heap / Priority Queue Operations",
    "Bitwise Operations & Masking",
    "Trie & String Processing",
]


def run_batch_test(items_per_paradigm: int = 2):
    print("=== STARTING PRODUCTION BATCH GENERATION & VERIFICATION PIPELINE ===")

    stats = {
        "generated": 0,
        "approved": 0,
        "dedup_dropped": 0,
        "quarantined": 0,
        "exceptions": 0,
    }

    os.makedirs("data", exist_ok=True)
    os.makedirs("data/quarantine", exist_ok=True)

    dedup_engine = CorpusBalanceEngine(max_per_paradigm=35, similarity_threshold=0.70)
    seed_counter = 42

    # -----------------------------------------------------------------------
    # Resume: pre-load previously approved entries so that a restart never
    # duplicates work or reuses already-processed seeds.
    # -----------------------------------------------------------------------
    approved_path = "data/approved_curriculum.jsonl"
    if os.path.exists(approved_path):
        loaded = 0
        max_seen_seed = seed_counter - 1
        with open(approved_path, "r", encoding="utf-8") as _f:
            for _line in _f:
                _line = _line.strip()
                if not _line:
                    continue
                try:
                    _entry = json.loads(_line)
                    _paradigm = _entry.get("paradigm", "")
                    _code = _entry.get("code", "")
                    _seed = _entry.get("seed", -1)
                    if _paradigm and _code:
                        dedup_engine.register_candidate(_paradigm, _code)
                        loaded += 1
                    if isinstance(_seed, int) and _seed > max_seen_seed:
                        max_seen_seed = _seed
                except (json.JSONDecodeError, KeyError):
                    pass
        # Place seed_counter past the highest seed already written to disk.
        seed_counter = max_seen_seed + 1
        print(f"[RESUME] Pre-loaded {loaded} existing approved entries. "
              f"Resuming from seed {seed_counter}.")
    # -----------------------------------------------------------------------

    for paradigm_idx, paradigm in enumerate(PARADIGMS):
        print(f"\n[+] Processing Paradigm [{paradigm_idx + 1}/{len(PARADIGMS)}]: '{paradigm}'...")

        for i in range(items_per_paradigm):
            sample_num = i + 1
            seed = seed_counter
            seed_counter += 1

            if not dedup_engine.can_accept_paradigm(paradigm):
                print(f"  [-] Sample {sample_num}: SKIPPED (Paradigm cap reached)")
                continue

            try:
                if i > 0 or paradigm_idx > 0:
                    time.sleep(2)

                success, candidate_data, result_msg = execute_retry_loop(
                    paradigm=paradigm,
                    seed=seed,
                    max_retries=3,
                )
                stats["generated"] += 1

                if not success or not candidate_data:
                    stats["quarantined"] += 1
                    print(f"  [X] Sample {sample_num}: QUARANTINED | Reason: {result_msg}")
                    quarantine_record = {
                        "paradigm": paradigm,
                        "seed": seed,
                        "reason": result_msg,
                        "status": "QUARANTINE",
                    }
                    with open(
                        f"data/quarantine/paradigm_{paradigm_idx}_quarantine.jsonl",
                        "a",
                        encoding="utf-8",
                    ) as f:
                        f.write(json.dumps(quarantine_record) + "\n")
                        f.flush()
                    continue

                full_code = candidate_data["code"]

                if dedup_engine.is_duplicate(full_code):
                    stats["dedup_dropped"] += 1
                    print(f"  [x] Sample {sample_num}: DROPPED (Duplicate AST structure)")
                    dup_record = {
                        "status": "QUARANTINE",
                        "reason": "Duplicate AST n-gram structure detected",
                        "candidate": candidate_data,
                    }
                    with open(
                        f"data/quarantine/paradigm_{paradigm_idx}_quarantine.jsonl",
                        "a",
                        encoding="utf-8",
                    ) as f:
                        f.write(json.dumps(dup_record) + "\n")
                        f.flush()
                    continue

                dedup_engine.register_candidate(paradigm, full_code)
                stats["approved"] += 1
                attempts = candidate_data.get("attempts_used", 1)

                with open("data/approved_curriculum.jsonl", "a", encoding="utf-8") as f:
                    f.write(json.dumps(candidate_data) + "\n")
                    f.flush()
                    os.fsync(f.fileno())

                print(f"  [OK] Sample {sample_num}: APPROVED | Attempts: {attempts}/3 | Seed: {seed}")

            except Exception as e:
                stats["exceptions"] += 1
                print(f"  [!] Exception on Paradigm '{paradigm}', Sample {sample_num}: {type(e).__name__} - {e}")
                continue

    total_valid = max(1, stats["generated"])
    print("\n" + "=" * 60)
    print("=== PRODUCTION BATCH PIPELINE TELEMETRY SUMMARY ===")
    print(f"  Total Generation Attempts:  {stats['generated'] + stats['exceptions']}")
    print(f"  Approved Solutions:         {stats['approved']} ({(stats['approved'] / total_valid) * 100:.1f}%)")
    print(f"  Quarantined / Hard Fails:   {stats['quarantined']} ({(stats['quarantined'] / total_valid) * 100:.1f}%)")
    print(f"  AST Topology Duplicates:    {stats['dedup_dropped']} ({(stats['dedup_dropped'] / total_valid) * 100:.1f}%)")
    print("=" * 60)


if __name__ == "__main__":
    run_batch_test(items_per_paradigm=1)
