# Verification-Guided Curriculum Learning (VGCL) Phase 0

## Project Overview

VGCL is an experimental framework for constructing synthetic training curricula targeted at the measured empirical failure modes of a base language model. Rather than generating synthetic training samples uniformly or via arbitrary topic heuristics, VGCL measures execution-grounded failure clusters of a fixed base model ($\theta_{\text{base}}$), synthesizes curriculum datasets designed to address those specific failure modes, and verifies candidate correctness through a two-phase differential verification oracle.

## Scientific Hypothesis

**Failure-targeted verified synthetic curricula produce greater capability transfer than uniformly sampled verified synthetic curricula under identical compute budgets.**

## Phase 0 Experimental Safeguards

The current Phase 0 implementation includes:

- deterministic failure attribution
- isolated oracle generation
- evaluation leakage checks
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
(targeted_gen.py)     (uniform_gen.py)
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

```bash
# 1. Base model failure attribution
python failure_mapper.py

# 2. Targeted curriculum generation
python targeted_generation.py --target-failure-cluster FC_01_MICRO_CORRECTNESS --num-samples 50

# 3. Uniform control generation
python uniform_generation.py --num-samples 50

# 4. Capability transfer analysis
python transfer_matrix.py

# 5. Statistical hypothesis evaluation
python statistical_analysis.py

# 6. Run full integration test suite
python -m unittest test_phase0_pipeline.py
```

## Experimental Controls

- **Targeted Pipeline (`targeted_generation.py`)**: Prompts candidate problems targeted specifically at empirical execution failure clusters (`FC_01`..`FC_08`).
- **Uniform Control Pipeline (`uniform_generation.py`)**: Identical generator model, decoupled 2-pass oracle, deduplication, ledger schema, and rate control, differing **only** by sampling uniformly across failure categories without failure targeting.

## Documentation & Auditing

- [Migration Guide](docs/MIGRATION.md)
- [Reproducibility Protocol](docs/REPRODUCIBILITY.md)
