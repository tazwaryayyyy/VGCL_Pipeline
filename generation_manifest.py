"""generation_manifest.py

Auditable Generation Manifest for VGCL Phase 0.

Produces and verifies deterministic experiment metadata manifests capturing:
- Git commit hash & repository freeze tag
- Generator model policy (primary + fallback sequence)
- Generation parameters (temperature, timeout, RPM limit, batch size)
- Quality gate thresholds (reference independence, deduplication)
- Pre-experiment calibration fixture hash
- Subtype allocation and uniform selection policies
- Random seed policies
"""

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional
import yaml


FROZEN_EXPERIMENT_CONFIG: Dict[str, Any] = {
    "freeze_tag": "phase0-experiment-freeze-v4",
    "generator_model": "gemini-3.6-flash",
    "fallback_policy": [
        "gemini-3.1-flash-lite",
        "gemini-3.5-flash-lite",
        "gemini-flash-latest",
    ],
    "rpm_limit": 15.0,
    "batch_size": 1,
    "generation_parameters": {
        "temperature": 0.4,
        "timeout_seconds": 30.0,
    },
    "prompt_version": "v1.0_decoupled_twopass",
    "syntax_retry_limit": 1,
    "reference_independence_threshold": 0.85,
    "reference_calibration_fixture_hash": "707762ab31720b7c",
    "quality_gate_versions": {
        "reference_independence": "0.85_ast_ngram",
        "subtype_validation": "types_a_f_ast_features_v1",
        "phase_a_scaling": "complexity_aware_adaptive_v1",
        "syntax_repair": "1shot_component_repair_v1",
    },
    "dedup_threshold": 0.70,
    "phase_a_policy_identifier": "complexity_aware_adaptive_v1",
    "targeted_subtype_allocation_policy": "stratified_strata_lowest_accepted_adaptive",
    "uniform_selection_policy": "round_robin_unstratified",
    "seed_policy": {
        "targeted_initial_seed": 1000,
        "uniform_initial_seed": 5000,
    },
}


def get_git_commit_hash() -> str:
    """Retrieves current git commit hash or 'unversioned_tree' if unavailable."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5.0,
        )
        return res.stdout.strip()
    except Exception:
        return "unknown_commit"


def load_yaml_config(config_path: str = "phase0_calibration/config.yaml") -> Dict[str, Any]:
    """Loads configuration from YAML file if present."""
    cfg_file = Path(config_path)
    if cfg_file.exists():
        with open(cfg_file, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def create_generation_manifest(
    config_path: str = "phase0_calibration/config.yaml",
    output_path: Optional[str] = "data/generation_manifest.json",
) -> Dict[str, Any]:
    """
    Constructs a deterministic manifest for the Phase 0 generation run.
    Merges disk configuration with frozen defaults and adds runtime commit provenance.
    """
    cfg = load_yaml_config(config_path)
    gen_cfg = cfg.get("generation_configuration", {})
    gen_pol = cfg.get("experimental_generator_policy", {})

    primary_model = gen_pol.get("primary_model", FROZEN_EXPERIMENT_CONFIG["generator_model"])
    fallback_models = gen_pol.get("fallback_models", FROZEN_EXPERIMENT_CONFIG["fallback_policy"])
    freeze_tag = gen_cfg.get("freeze_tag", FROZEN_EXPERIMENT_CONFIG["freeze_tag"])

    manifest: Dict[str, Any] = {
        "code_commit": get_git_commit_hash(),
        "freeze_tag": freeze_tag,
        "generator_model": primary_model,
        "fallback_policy": fallback_models,
        "rpm_limit": gen_cfg.get("rpm_limit", FROZEN_EXPERIMENT_CONFIG["rpm_limit"]),
        "batch_size": gen_cfg.get("batch_size", FROZEN_EXPERIMENT_CONFIG["batch_size"]),
        "generation_parameters": {
            "temperature": gen_pol.get("temperature", FROZEN_EXPERIMENT_CONFIG["generation_parameters"]["temperature"]),
            "timeout_seconds": gen_pol.get("timeout_seconds", FROZEN_EXPERIMENT_CONFIG["generation_parameters"]["timeout_seconds"]),
        },
        "prompt_version": FROZEN_EXPERIMENT_CONFIG["prompt_version"],
        "syntax_retry_limit": gen_cfg.get("syntax_retry_limit", FROZEN_EXPERIMENT_CONFIG["syntax_retry_limit"]),
        "reference_independence_threshold": gen_cfg.get(
            "reference_independence_threshold",
            FROZEN_EXPERIMENT_CONFIG["reference_independence_threshold"],
        ),
        "reference_calibration_fixture_hash": gen_cfg.get(
            "reference_calibration_fixture_hash",
            FROZEN_EXPERIMENT_CONFIG["reference_calibration_fixture_hash"],
        ),
        "quality_gate_versions": FROZEN_EXPERIMENT_CONFIG["quality_gate_versions"],
        "dedup_threshold": gen_cfg.get("dedup_threshold", FROZEN_EXPERIMENT_CONFIG["dedup_threshold"]),
        "phase_a_policy_identifier": gen_cfg.get(
            "phase_a_policy_identifier",
            FROZEN_EXPERIMENT_CONFIG["phase_a_policy_identifier"],
        ),
        "targeted_subtype_allocation_policy": gen_cfg.get(
            "targeted_subtype_allocation_policy",
            FROZEN_EXPERIMENT_CONFIG["targeted_subtype_allocation_policy"],
        ),
        "uniform_selection_policy": gen_cfg.get(
            "uniform_selection_policy",
            FROZEN_EXPERIMENT_CONFIG["uniform_selection_policy"],
        ),
        "random_seed_policy": gen_cfg.get("seed_policy", FROZEN_EXPERIMENT_CONFIG["seed_policy"]),
    }

    if output_path:
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

    return manifest


if __name__ == "__main__":
    manifest = create_generation_manifest()
    print("=== VGCL PHASE 0 GENERATION MANIFEST ===")
    print(json.dumps(manifest, indent=2))
