# VGCL Phase 0: Reproducibility Protocol & External Audit Specification

This repository implements a **causally identifiable verification-guided curriculum learning (VGCL) experimental framework** designed for rigorous scientific verification, audit, and replication.

Central Hypothesis:
> **Failure-targeted verified synthetic curricula produce greater capability transfer than uniformly sampled verified synthetic curricula under identical compute budgets.**

---

## I. Non-Negotiable Experimental Invariants

1. **Causal Targeting**: Every curriculum sample in the targeted condition MUST explicitly map to an empirically measured failure cluster of the base model ($\theta_{\text{base}}$).
2. **Reproducible Verification**: Every acceptance or rejection decision MUST be deterministically reproducible from stored artifacts in `data/quarantine/` and `data/experiment_ledger.db`.
3. **Oracle Independence**: Brute-force reference implementations MUST be generated using isolated prompt strategies in a separate LLM context window with strict algorithmic constraints (exhaustive search, no memoization, no advanced data structures).
4. **Structural Novelty**: Accepted curriculum samples MUST satisfy $S_{\text{AST/CFG}} < 0.70$ against all existing problems in the corpus.
5. **Controlled Comparison**: The targeted and uniform control pipelines differ **ONLY** in curriculum targeting strategy. Generator model, prompts, verifier, deduplication, ledger schema, rate control, fine-tuning hyperparameters, and evaluation suites are identical.
6. **Statistical Validity**: Evaluation is cluster-level, pre-registered, and reported as $\mu \pm \sigma$ across 3 fixed random seeds (Seeds: 42, 1337, 2026).
7. **Dataset Separation**: Calibration samples are drawn strictly from disjoint data splits (`google-research-datasets/mbpp`, config: `sanitized`, split: `train`), guaranteeing $\text{Calibration} \cap \text{Training} \cap \text{Evaluation} = \emptyset$. Benchmark evaluation sets (HumanEval, MBPP `test`) are strictly excluded from calibration. Artifact versioning enforces retirement of legacy contaminated calibration datasets to `data/archive/contaminated_calibration_v1/`.

---

## II. Module Reference & Reproducibility Pipeline

```
┌─────────────────────────┐
│ failure_mapper.py       │ ──> Probes base model & constructs deterministic failure clusters FC_01..FC_08
└─────────────────────────┘
             │
             ├───────────────────────────────┐
             ▼                               ▼
┌─────────────────────────┐     ┌─────────────────────────┐
│ targeted_generation.py  │     │ uniform_generation.py   │ ──> Generates targeted vs uniform control datasets
└─────────────────────────┘     └─────────────────────────┘
             │                               │
             └───────────────┬───────────────┘
                             ▼
                ┌─────────────────────────┐
                │ oracle_refactor.py      │ ──> Two-phase differential oracle & permanent quarantine
                └─────────────────────────┘
                             │
                             ▼
                ┌─────────────────────────┐
                │ cfg_dedup.py            │ ──> AST + CFG structural deduplication (< 0.70 threshold)
                └─────────────────────────┘
                             │
                             ▼
                ┌─────────────────────────┐
                │ experiment_ledger.py    │ ──> Immutable SQLite experiment ledger
                └─────────────────────────┘
                             │
                             ▼
                ┌─────────────────────────┐
                │ transfer_matrix.py      │ ──> Computes M_{k,j} cluster transfer matrix
                └─────────────────────────┘
                             │
                             ▼
                ┌─────────────────────────┐
                │ statistical_analysis.py │ ──> Hypothesis testing (Welch's t-test, Cohen's d, CI)
                └─────────────────────────┘
```

---

## III. Pre-Registered Training Protocol (LoRA Fine-Tuning)

All model fine-tuning experiments MUST adhere to the following pre-registered hyperparameter configuration:

- **Base Model**: `Qwen/Qwen2.5-1.5B-Instruct`
- **LoRA Rank ($r$)**: `16`
- **LoRA Alpha ($\alpha$)**: `32`
- **LoRA Target Modules**: `["q_proj", "k_proj", "v_proj", "o_proj"]`
- **Learning Rate**: `2e-4`
- **Epochs**: `1`
- **Batch Size**: `8` (Micro-batch: 2, Gradient Accumulation: 4)
- **Optimizer**: `AdamW` ($\beta_1=0.9, \beta_2=0.999, \epsilon=1\times 10^{-8}$)
- **Weight Decay**: `0.01`
- **LR Scheduler**: `Cosine` with 5% warmup
- **Random Seeds**: `[42, 1337, 2026]`

---

## IV. Audit Commands for External Researchers

1. **Verify Failure Attribution & Clustering**:
   ```bash
   python failure_mapper.py
   ```
2. **Audit Verification Oracle & Quarantine Records**:
   ```bash
   python -c "from oracle_refactor import VerificationOracle; o = VerificationOracle(); print('Quarantine dir:', o.quarantine_dir)"
   ```
3. **Audit Structural Novelty & CFG Hashing**:
   ```bash
   python cfg_dedup.py
   ```
4. **Inspect Immutable Experiment Ledger**:
   ```bash
   python -c "from experiment_ledger import ExperimentLedger; l = ExperimentLedger(); print('Ledger entries:', len(l.get_entries()))"
   ```
5. **Reproduce Statistical Analysis & Hypothesis Report**:
   ```bash
   python statistical_analysis.py
   ```
