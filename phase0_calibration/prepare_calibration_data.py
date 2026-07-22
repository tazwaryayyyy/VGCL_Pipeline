import json
import re
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer

# Load tokenizer to accurately determine target token IDs for Qwen 2.5
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def extract_first_token(code_solution: str) -> str:
    """Extracts the first non-whitespace representation token from a reference solution."""
    stripped = code_solution.strip()
    if not stripped:
        return "return"

    # Tokenize solution and extract the first token string
    tokens = tokenizer.tokenize(stripped)
    if tokens:
        # Convert token format (handling byte-fallback/prefix characters)
        raw_token = tokenizer.convert_tokens_to_string([tokens[0]]).strip()
        return raw_token if raw_token else "def"
    return "def"


def assign_heuristic_category(code_text: str) -> int:
    """Categorizes problems into 1 of 8 heuristic taxonomy buckets (0-7)."""
    text = code_text.lower()
    if any(k in text for k in ["for ", "while ", "enumerate"]):
        return 0  # Iteration / Loops
    elif any(k in text for k in ["if ", "elif ", "else:"]):
        return 1  # Control Flow / Conditionals
    elif any(k in text for k in ["math.", "sqrt", "pow", "%", "abs("]):
        return 2  # Mathematical Operations
    elif any(k in text for k in ["split", "join", "replace", "strip", "substring"]):
        return 3  # String Processing
    elif any(k in text for k in ["append", "pop", "sort", "dict", "set("]):
        return 4  # Data Structures
    elif any(k in text for k in ["def ", "return "]):
        return 5  # Functional Logic
    elif any(k in text for k in ["try:", "except", "isinstance"]):
        return 6  # Type Checking / Defensiveness
    else:
        return 7  # General Algorithmic Logic


def build_calibration_dataset(humaneval_path: str, output_path: str, target_count=500):
    calibration_items = []

    # 1. Process HumanEval
    print(f"Loading HumanEval from {humaneval_path}...")
    with open(humaneval_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            sol = item.get("canonical_solution", "")

            calibration_items.append({
                "source": "HumanEval",
                "task_id": item["task_id"],
                "prompt": item["prompt"],
                "target_token": extract_first_token(sol),
                "human_category_id": assign_heuristic_category(sol),
                "canonical_solution": sol
            })

    print(f"Loaded {len(calibration_items)} problems from HumanEval.")

    # 2. Process MBPP from Hugging Face
    needed = target_count - len(calibration_items)
    print(f"Fetching MBPP dataset to fulfill remaining {needed} problems...")

    # Load sanitized MBPP split
    mbpp_ds = load_dataset("google-research-datasets/mbpp",
                           "sanitized", split="test")

    for idx, item in enumerate(mbpp_ds):
        sol = item.get("code", "")
        calibration_items.append({
            "source": "MBPP",
            "task_id": f"MBPP/{item['task_id']}",
            "prompt": item["prompt"],
            "target_token": extract_first_token(sol),
            "human_category_id": assign_heuristic_category(sol),
            "canonical_solution": sol
        })

        if len(calibration_items) >= target_count:
            break

    # Save finalized standard dataset
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(calibration_items, f, indent=2)

    print(
        f"Successfully generated {len(calibration_items)} calibration problems at: {out_file.resolve()}")


if __name__ == "__main__":
    # Adjust path if your human eval file is named differently (e.g. human-eval.jsonl)
    build_calibration_dataset(
        humaneval_path="data/human-eval.jsonl",
        output_path="data/raw_calibration.json",
        target_count=500
    )
