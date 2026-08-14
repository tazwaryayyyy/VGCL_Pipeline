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

## II. Throughput Optimization Semantics

The throughput optimization is engineering-only and does not change oracle thresholds, failure taxonomy, AST/CFG deduplication, ledger schema, calibration data, evaluation data, or the two-pass oracle independence design.

`--batch-size N` means that up to `N` independent candidate generation tasks are submitted in one outer loop iteration. A successful candidate still performs the same two isolated LLM passes: Pass 1 generates `PROBLEM_SPEC`, `SOLVE_OPTIMIZED`, and `TEST_GENERATOR`; Pass 2 receives only `PROBLEM_SPEC` and generates `SOLVE_BRUTE`. Therefore `batch_size=3` does not mean one Gemini request returning three candidates. It means up to three concurrent candidate workers, normally requiring up to six successful Gemini requests in total.

Concurrency is limited to candidate generation. Verification oracle execution, syntax validation, structural deduplication, calibration, and ledger writes are processed after generation results are collected. Results are consumed in sorted seed order, so completion order cannot change acceptance order, deduplication order, or scheduler feedback order.

Seeds are assigned before generation workers are submitted. This provides seed determinism and deterministic result ordering. It does not imply that adaptive-policy trajectories are identical for every `batch_size`: when `batch_size > 1`, subtype choices for the batch are made before observing within-batch outcomes. The targeted scheduler therefore reserves planned accepted-count slots before dispatch and restricts approval-weighted sampling to subtypes tied for the lowest accepted count. Approval feedback can change attempt ordering within tied strata, but it must not let easy-to-generate subtypes overtake the accepted curriculum allocation.

The pre-oracle syntax stage is a normalization gate, not a semantic repair system. It validates `ast.parse()` in process, strips leading/trailing markdown code fences and trailing whitespace when needed, and revalidates. It does not change algorithmic logic, identifiers, control flow, or oracle criteria. If normalization fails, the candidate is rejected before oracle submission.

Metrics are defined as follows:

- `approval_rate = approved / generation_attempts`
- `syntax_pass_rate = syntax_passes / syntax_attempts`, where `syntax_attempts = generation_attempts - generation_block_parse_errors`
- `phase_a_pass_rate = phase_a_passes / oracle_submissions`
- `phase_b_pass_rate = phase_b_passes / phase_b_submissions`, where `phase_b_submissions = phase_a_passes`
- `dedup_rejection_rate = dedup_rejections / dedup_submissions`, where `dedup_submissions = phase_b_passes`
- `api_calls_per_approved = api_calls / approved`
- `approval_rate_by_subtype = approved_subtype / attempted_subtype`
- `approval_rate_by_cluster = approved_cluster / attempted_cluster`

Throughput benchmarks must use disposable outputs and report actual measured wall-clock time, API calls, attempts, pass rates, deduplication rejections, accepted candidates, approved candidates per minute, approved candidates per API call, and mean generation/verification latencies. Mock benchmarks may validate local batching mechanics, but they are not evidence of real Gemini API throughput.

---

## III. Module Reference & Reproducibility Pipeline

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

## IV. Pre-Registered Training Protocol (LoRA Fine-Tuning)

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

## V. Audit Commands for External Researchers

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
