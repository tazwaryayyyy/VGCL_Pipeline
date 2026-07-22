import os
import json
import glob
from pathlib import Path
from verifier import evaluate_and_route


def load_items_from_file(file_path: str) -> list:
    """Safely extracts JSON records from single JSON objects, arrays, or JSONL lines."""
    items = []
    if not os.path.exists(file_path):
        return items

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read().strip()
        if not content:
            return items

        # Try parsing as JSON array/object
        try:
            parsed = json.loads(content)
            if isinstance(parsed, list):
                return [json.dumps(x) for x in parsed]
            elif isinstance(parsed, dict):
                return [json.dumps(parsed)]
        except json.JSONDecodeError:
            pass

        # Fallback: Parse line-by-line (JSONL format)
        for line in content.splitlines():
            line = line.strip()
            if line:
                items.append(line)

    return items


def run_salvage_pipeline(
    review_queue_path: str = "review_queue.json",
    quarantine_dir: str = "quarantine",
    approved_output_path: str = "approved_curriculum.jsonl",
):
    print("=== STARTING PHASE 2: SALVAGE & AUDIT PIPELINE ===")

    raw_candidates = []

    # 1. Collect from review_queue.json
    if os.path.exists(review_queue_path):
        review_items = load_items_from_file(review_queue_path)
        print(
            f"[+] Loaded {len(review_items)} candidates from {review_queue_path}")
        raw_candidates.extend(review_items)
    else:
        print(f"[-] Warning: {review_queue_path} not found. Skipping.")

    # 2. Collect from quarantine directory
    if os.path.exists(quarantine_dir):
        q_files = glob.glob(os.path.join(
            quarantine_dir, "**", "*"), recursive=True)
        q_count = 0
        for qf in q_files:
            if os.path.isfile(qf) and (
                qf.endswith(".json") or qf.endswith(
                    ".jsonl") or qf.endswith(".txt")
            ):
                items = load_items_from_file(qf)
                raw_candidates.extend(items)
                q_count += len(items)
        print(
            f"[+] Loaded {q_count} candidates from quarantine files in '{quarantine_dir}/'")
    else:
        print(
            f"[-] Warning: Quarantine folder '{quarantine_dir}' not found. Skipping.")

    if not raw_candidates:
        print("[!] No candidates found to evaluate. Exiting.")
        return

    print(
        f"\n[*] Processing {len(raw_candidates)} total candidates through hardened verifier...\n")

    approved_list = []
    review_queue_list = []
    quarantine_list = []

    for raw_str in raw_candidates:
        res = evaluate_and_route(raw_str)
        status = res["status"]

        if status == "APPROVED":
            approved_list.append(res["data"])
        elif status == "REVIEW_QUEUE":
            review_queue_list.append(res)
        else:
            quarantine_list.append(res)

    # 3. Write recovered items to approved_curriculum.jsonl (Append mode)
    if approved_list:
        with open(approved_output_path, "a", encoding="utf-8") as f:
            for item in approved_list:
                f.write(json.dumps(item) + "\n")
        print(
            f"[SUCCESS] Appended {len(approved_list)} salvaged items to {approved_output_path}")

    # 4. Overwrite review_queue.json with remaining borderline cases
    with open(review_queue_path, "w", encoding="utf-8") as f:
        json.dump(review_queue_list, f, indent=2)
    print(
        f"[UPDATE] Rewrote {review_queue_path} with {len(review_queue_list)} verified borderline items")

    # 5. Save categorized failure diagnostics for quarantine items
    quarantine_log_path = "quarantine_audit.jsonl"
    with open(quarantine_log_path, "w", encoding="utf-8") as f:
        for q_item in quarantine_list:
            f.write(json.dumps(q_item) + "\n")
    print(
        f"[DIAGNOSTICS] Saved {len(quarantine_list)} hard-failure items to {quarantine_log_path}")

    # Summary
    print("\n=== SALVAGE SUMMARY ===")
    print(f"  • Approved & Recovered: {len(approved_list)}")
    print(f"  • Retained in Review Queue: {len(review_queue_list)}")
    print(f"  • Confirmed Irrecoverable/Quarantined: {len(quarantine_list)}")


if __name__ == "__main__":
    run_salvage_pipeline(
        review_queue_path="data/review_queue.jsonl",
        quarantine_dir="data/quarantine",
        approved_output_path="data/approved_curriculum.jsonl"
    )
