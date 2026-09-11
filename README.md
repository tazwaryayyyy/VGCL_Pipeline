# Verification-Guided Curriculum Learning (VGCL) Phase 0

## Project Overview

VGCL is an experimental framework for constructing synthetic training curricula targeted at the measured empirical failure modes of a base language model. Rather than generating synthetic training samples uniformly or via arbitrary topic heuristics, VGCL measures execution-grounded failure clusters of a fixed base model ($\theta_{\text{base}}$), synthesizes curriculum datasets designed to address those specific failure modes, and verifies candidate correctness through a two-phase differential verification oracle.

## Scientific Hypothesis

**Failure-targeted verified synthetic curricula produce greater capability transfer than uniformly sampled verified synthetic curricula under identical compute budgets.**

## Phase 0 Experimental Safeguards

The current Phase 0 implementation includes:

- deterministic failure attribution
- isolated oracle generation
- strict evaluation leakage checks ($\text{Calibration} \cap \text{Training} \cap \text{Evaluation} = \emptyset$)
- explicit dataset configuration (`phase0_calibration/config.yaml`) using MBPP `train` for calibration
- legacy contaminated calibration artifacts safely retired (`data/archive/contaminated_calibration_v1/`)
- cluster exposure auditing
- matched targeted and uniform control pipelines

This repository version is frozen for Phase 0 experiments.

## Pipeline Diagram

```
Base Model Probing (failure_mapper.py)
            │
            ▼
Deterministic Failure Attribution (FC_01...FC_08)
            │
  ┌─────────┴─────────┐
  ▼                   ▼
Targeted Generation   Uniform Control Generation
(targeted_generation.py) (uniform_generation.py)
  │                   │
  └─────────┬─────────┘
            ▼
Two-Phase Verification Oracle (oracle_refactor.py)
            │
            ▼
Structural Deduplication AST + CFG (cfg_dedup.py)
            │
            ▼
Experiment Ledger (experiment_ledger.py)
            │
            ▼
Cluster-Level Transfer Matrix (transfer_matrix.py)
            │
            ▼
Statistical Analysis & Significance Testing (statistical_analysis.py)
```

## Installation

```bash
git clone https://github.com/your-username/vgcl-pipeline.git
cd vgcl-pipeline
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Set environment variable in `.env`:
```env
GEMINI_API_KEY=your_api_key
```

## Running Phase 0

### Workflow Overview (Local Generation + Remote Kaggle Probing)

To support memory- and disk-constrained local development environments without compromising scientific rigor, candidate generation and verification run locally (with a strict local guard preventing `Qwen/Qwen2.5-1.5B-Instruct` downloads), while empirical base-model probing runs remotely on Kaggle GPUs:

```bash
# 1. Local Candidate Generation & Oracle Verification (Targeted)
# Generates, verifies, deduplicates, and writes candidates to data/v7_probe_manifest.jsonl
python targeted_generation.py --target-failure-cluster FC_01_MICRO_CORRECTNESS --num-samples 50 --mode local

# 2. Local Candidate Generation & Oracle Verification (Uniform Control)
# Generates uniform control candidates and writes to data/v7_probe_manifest.jsonl
python uniform_generation.py --num-samples 50 --mode local

# 3. Remote Base-Model Probing (Executed on Kaggle GPU)
# Runs real Qwen/Qwen2.5-1.5B-Instruct Pass@1 / Pass@10 execution probing
python kaggle_probe.py --manifest data/v7_probe_manifest.jsonl --output data/v7_kaggle_probe_results.jsonl

# 4. Local Ingestion & Primary Eligibility Merge
# Verifies cryptographic provenance, applies V7 primary eligibility gates, and updates ledger
python merge_probe_results.py --probes data/v7_kaggle_probe_results.jsonl

# 5. Capability transfer analysis
python transfer_matrix.py

# 6. Statistical hypothesis evaluation
python statistical_analysis.py

# 7. Run full integration test suite (77 tests)
pytest -v test_phase0_pipeline.py
```

## Experimental Controls

- **Targeted Pipeline (`targeted_generation.py`)**: Prompts candidate problems targeted specifically at empirical execution failure clusters (`FC_01`..`FC_08`).
- **Uniform Control Pipeline (`uniform_generation.py`)**: Identical generator model, decoupled 2-pass oracle, deduplication, ledger schema, and rate control, differing **only** by sampling uniformly across failure categories without failure targeting.

## Generator Model Policy (Frozen at phase0-experiment-freeze-v7)

Both pipelines use the same explicit `EXPERIMENTAL_GENERATOR_POLICY`:

| Parameter | Value |
|---|---|
| Primary model | `gemini-3.6-flash` |
| Fallback sequence | `gemini-3.1-flash-lite` → `gemini-3.5-flash-lite` → `gemini-flash-latest` |
| Temperature | `0.4` |
| Rate limit | `15 RPM` |

- Fallback candidates are strictly quarantined to auxiliary datasets (`data/v7_auxiliary_curriculum.jsonl`, `data/v7_uniform_auxiliary_curriculum.jsonl`) and **never enter the primary curriculum dataset**.
- Both pipelines import `EXPERIMENTAL_GENERATOR_POLICY` and `_MODEL_PRIORITY` from `targeted_generation.py` — verified by test suites.

## Reference-Independence Threshold (Frozen at 0.85)

The 0.85 threshold was validated by a pre-experiment calibration fixture (`phase0_calibration/reference_fixture.py`) before any curriculum was generated:

- **Fixture hash**: `707762ab31720b7c`
- **Result**: 11/11 assertions passed. All duplicate/near-duplicate pairs → REJECT (scores 1.0). All structurally distinct pairs → PASS (scores ≤ 0.05). No borderline ambiguity.
- **Fixture independence**: Synthetic engineering pairs only — independent of MBPP, HumanEval, and any curriculum data.

## Auditable Generation Manifest

Before generation, `generation_manifest.py` writes `data/generation_manifest.json` capturing: commit hash, freeze tag, generator model, fallback policy, temperature, RPM limit, batch size, quality-gate versions, thresholds, calibration fixture hash, and seed policies.

## Throughput Optimization & Quality Gates Status

Both generation pipelines accept `--batch-size N`. `N` is the number of candidate generation tasks dispatched per outer loop iteration. Each candidate still uses the same decoupled two-pass LLM generation design, so a successful candidate normally requires two Gemini API requests. Generation tasks are run concurrently with `ThreadPoolExecutor`; verification, deduplication, calibration, and ledger writes are consumed afterward in deterministic seed order.

The targeted subtype scheduler uses approval feedback only among subtypes with the lowest accepted count, with planned batch slots reserved before dispatch. This preserves the intended accepted subtype allocation while allowing adaptive attempt ordering within tied strata.

### Four Surgical Scientific Quality Gates (Phase 0)
1. **Optimized/Reference Structural Independence Gate (`quality_gates.py`)**: Checks whether the generated optimized solution (`solve`) and brute-force reference (`solve_brute`) are suspiciously isomorphic ($S_{\text{AST}} \ge 0.85$ or identical CFG/AST). Rejects and quarantines near-isomorphic pairs before verification.
2. **Subtype Validity Gate (`quality_gates.py`)**: Applies deterministic AST/structural evidence verification to ensure candidates genuinely instantiate declared curriculum subtypes (Types A–F). Mismatched candidates are quarantined.
3. **Complexity-Aware Phase A Scaling (`oracle_refactor.py`)**: Derives conservative Phase A test scale bounds ($N \le 5$) for combinatorial/recursive brute-force solvers, preventing timeouts while maintaining strict differential verification.
4. **Controlled 1-Shot Syntax Failure Regeneration (`syntax_repair.py`)**: If generated code fails Python syntax parsing, pinpoints the malformed component (`solve_opt`, `solve_brute`, or `test_gen`) and performs at most one isolated component-level regeneration attempt. Spec and other components remain unchanged.

## Documentation & Auditing

- [Migration Guide](docs/MIGRATION.md)
- [Reproducibility Protocol](docs/REPRODUCIBILITY.md)
- [Experiment Freeze](docs/EXPERIMENT_FREEZE.md)
