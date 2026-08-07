import glob
import json
import os
from pathlib import Path
from typing import Any, Dict

from verifier import evaluate_and_route


def atomic_write_json(file_path: str, data: Any) -> None:
    """Atomic write primitive preventing zero-byte file corruption during interruptions."""
    temp_path = f"{file_path}.tmp"
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp_path, file_path)


def normalize_to_canonical_schema(item: Dict[str, Any]) -> Dict[str, Any]:
    """Coerces arbitrary candidate dicts into the standard canonical schema."""
    code = item.get("code", "")
    solve_opt = item.get("solve_opt", "")
    solve_brute = item.get("solve_brute", "")
    test_gen = item.get("test_gen", "")

    if not code and solve_opt:
        code = f"{solve_opt}\n\n{solve_brute}\n\n{test_gen}".strip()

    return {
        "paradigm": item.get("paradigm", item.get("cluster_id", "Unassigned")),
        "seed": item.get("seed", 42),
        "spec": item.get("spec", item.get("problem", "")),
        "code": code,
        "solve_opt": solve_opt,
        "solve_brute": solve_brute,
        "test_gen": test_gen,
        "attempts_used": item.get("attempts_used", 1),
    }


def load_items_from_file(file_path: str) -> list:
    """Safely extracts JSON records from single JSON objects, arrays, or JSONL lines."""
    items = []
    if not os.path.exists(file_path):
        return items

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read().strip()
        if not content:
            return items

        try:
            parsed = json.loads(content)
            if isinstance(parsed, list):
                return [json.dumps(x) for x in parsed]
            elif isinstance(parsed, dict):
                return [json.dumps(parsed)]
        except json.JSONDecodeError:
            pass

        for line in content.splitlines():
            line = line.strip()
            if line:
                items.append(line)

    return items


def run_salvage_pipeline(
    review_queue_path: str = "data/review_queue.json",
    quarantine_dir: str = "data/quarantine",
    approved_output_path: str = "data/approved_curriculum.jsonl",
):
    print("=== STARTING PHASE 2: SALVAGE & AUDIT PIPELINE ===")

    raw_candidates = []

    if os.path.exists(review_queue_path):
        review_items = load_items_from_file(review_queue_path)
        print(f"[+] Loaded {len(review_items)} candidates from {review_queue_path}")
        raw_candidates.extend(review_items)

    if os.path.exists(quarantine_dir):
        q_files = glob.glob(os.path.join(quarantine_dir, "**", "*"), recursive=True)
        q_count = 0
        for qf in q_files:
            if os.path.isfile(qf) and (qf.endswith(".json") or qf.endswith(".jsonl") or qf.endswith(".txt")):
                items = load_items_from_file(qf)
                raw_candidates.extend(items)
                q_count += len(items)
        print(f"[+] Loaded {q_count} candidates from quarantine files in '{quarantine_dir}/'")

    if not raw_candidates:
        print("[!] No candidates found to evaluate. Exiting.")
        return

    approved_list = []
    review_queue_list = []
    quarantine_list = []

    for raw_str in raw_candidates:
        res = evaluate_and_route(raw_str)
        status = res["status"]

        if status == "APPROVED":
            approved_list.append(normalize_to_canonical_schema(res["data"]))
        elif status == "REVIEW_QUEUE":
            review_queue_list.append(res)
        else:
            quarantine_list.append(res)

    if approved_list:
        with open(approved_output_path, "a", encoding="utf-8") as f:
            for item in approved_list:
                f.write(json.dumps(item) + "\n")
            f.flush()
            os.fsync(f.fileno())
        print(f"[SUCCESS] Appended {len(approved_list)} salvaged items to {approved_output_path}")

    atomic_write_json(review_queue_path, review_queue_list)
    print(f"[UPDATE] Rewrote {review_queue_path} with {len(review_queue_list)} verified items")

    quarantine_log_path = "data/quarantine_audit.jsonl"
    with open(quarantine_log_path, "w", encoding="utf-8") as f:
        for q_item in quarantine_list:
            f.write(json.dumps(q_item) + "\n")
        f.flush()
        os.fsync(f.fileno())
    print(f"[DIAGNOSTICS] Saved {len(quarantine_list)} hard-failure items to {quarantine_log_path}")


if __name__ == "__main__":
    run_salvage_pipeline()
