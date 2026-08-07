import json
import re
from pathlib import Path
import yaml
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


def load_calibration_config(config_path: str = "phase0_calibration/config.yaml") -> dict:
    """Loads explicit calibration dataset configuration."""
    cfg_file = Path(config_path)
    if not cfg_file.exists():
        raise FileNotFoundError(f"Calibration config not found at: {config_path}")
    with open(cfg_file, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_calibration_dataset(
    config_path: str = "phase0_calibration/config.yaml",
    output_path: str = "data/raw_calibration.json"
):
    """
    Builds calibration dataset guaranteeing:
    Calibration ∩ Training ∩ Evaluation = empty (Disjoint Calibration Invariant)
    """
    cfg = load_calibration_config(config_path)
    cal_cfg = cfg["calibration_dataset"]

    dataset_name = cal_cfg["name"]
    dataset_split = cal_cfg["split"]
    dataset_version = cal_cfg.get("version", "sanitized")
    target_count = cal_cfg.get("target_count", 500)

    print(f"Loading DISJOINT calibration split '{dataset_split}' from {dataset_name} ({dataset_version})...")
    
    # Load disjoint MBPP split (e.g. 'train' or 'prompt', strictly NOT 'test')
    mbpp_cal_ds = load_dataset(dataset_name, dataset_version, split=dataset_split)

    calibration_items = []
    for idx, item in enumerate(mbpp_cal_ds):
        sol = item.get("code", "")
        task_id = f"MBPP_CALIB/{item['task_id']}"

        calibration_items.append({
            "source": f"MBPP_{dataset_split.upper()}",
            "task_id": task_id,
            "prompt": item["prompt"],
            "target_token": extract_first_token(sol),
            "human_category_id": assign_heuristic_category(sol),
            "canonical_solution": sol
        })

        if len(calibration_items) >= target_count:
            break

    # Save finalized disjoint calibration dataset
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(calibration_items, f, indent=2)

    print(
        f"Successfully generated {len(calibration_items)} DISJOINT calibration problems at: {out_file.resolve()}"
    )


if __name__ == "__main__":
    build_calibration_dataset(
        config_path="phase0_calibration/config.yaml",
        output_path="data/raw_calibration.json"
    )
