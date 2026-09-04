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

The pre-oracle syntax stage is a normalization gate, not a semantic repair system. It validates `ast.parse()` in process, strips leading/trailing markdown code fences and trailing whitespace when needed, and revalidates. It does not change algorithmic logic, identifiers, control flow, or oracle criteria. If normalization fails, the candidate undergoes at most one component-level syntax regeneration retry. If it still fails, it is quarantined.

## II.A. Four Surgical Scientific Quality Gates

1. **Optimized/Reference Structural Independence Gate (`quality_gates.py`)**:
   - **Method**: Isolated normalized AST n-gram token similarity between `solve` and `solve_brute`.
   - **Threshold**: Deterministic $0.85$ near-isomorphism threshold (stricter than dataset deduplication $0.70$ because both functions address the same problem specification).
   - **Rejection Action**: Candidates with identical AST/CFG shape or similarity $\ge 0.85$ are quarantined with full diagnostic provenance without oracle execution.
2. **Subtype Validity Gate (`quality_gates.py`)**:
   - **Method**: Deterministic AST structural feature checks (data structures, loop depths, recursion, operation patterns) validating that candidate problems instantiate declared curriculum subtypes (`Type A` through `Type F`).
   - **Rejection Action**: Candidates lacking required structural features are quarantined (`status: REJECT`). Unrecognized subtypes return `NOT_VALIDATABLE` without mutating candidates.
3. **Complexity-Aware Phase A Scaling (`oracle_refactor.py`)**:
   - **Method**: Derives dimension limits from observable brute-force solver AST features.
   - **Scale Policy**: Pathological brute solvers (recursion, loop depth $\ge 3$, or `itertools` combinatorial calls) receive a conservative bound ($N \le 5$, `pathological_bruteforce_cap`). Standard solvers retain $N \le 8$ (`default_small_scale`).
   - **Invariant**: Preserves differential verification without skipping Phase A or inflating timeouts.
4. **Controlled 1-Shot Syntax Component Regeneration (`syntax_repair.py`)**:
   - **Method**: Pinpoints the single malformed component (`solve_opt`, `solve_brute`, or `test_gen`) and executes at most one targeted regeneration prompt.
   - **Constraints**: Problem spec, target cluster, subtype, seed lineage, and valid components are immutable.
   - **Accounting**: Distinguishes `initial_generation_api_calls` and `syntax_retry_api_calls`. Total API calls = initial + retries.

Metrics are defined as follows:

- `approval_rate = approved / generation_attempts`
- `syntax_pass_rate = syntax_passes / syntax_attempts`, where `syntax_attempts = generation_attempts - generation_block_parse_errors`
- `phase_a_pass_rate = phase_a_passes / oracle_submissions`
- `phase_b_pass_rate = phase_b_passes / phase_b_submissions`, where `phase_b_submissions = phase_a_passes`
- `dedup_rejection_rate = dedup_rejections / dedup_submissions`, where `dedup_submissions = phase_b_passes`
- `api_calls_per_approved = api_calls / approved`
- `approval_rate_by_subtype = approved_subtype / attempted_subtype`
- `approval_rate_by_cluster = approved_cluster / attempted_cluster`

Granular diagnostic metric counters include:
- `generation_attempts`
- `initial_generation_api_calls`
- `syntax_retry_api_calls`
- `total_generation_api_calls`
- `syntax_failures`
- `syntax_retry_successes`
- `reference_independence_rejections`
- `subtype_rejections`
- `phase_a_failures`
- `phase_b_failures`
- `dedup_rejections`
- `approved`

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

---

## VI. Generator Model Policy, Threshold Calibration & Reproducibility Limitations

### Generator Model Policy

The main experiment uses an explicit, auditable generator policy declared in `EXPERIMENTAL_GENERATOR_POLICY` (sourced from `targeted_generation.py`, shared by `uniform_generation.py`):

- **Primary model**: `gemini-3.6-flash`
- **Fallback sequence** (free-tier quota failover only): `gemini-3.1-flash-lite` → `gemini-3.5-flash-lite` → `gemini-flash-latest`
- **Temperature**: `0.4`
- **Client timeout**: `30.0s`
- **Auditability**: Every API call returns the actual serving model name. That model name is recorded in all quarantine artifacts and in the `generator_model` field of every ledger entry. Fallback transitions are logged to stdout with `[FALLBACK-ROUTING]`.
- **Symmetry**: Targeted and uniform generation import `EXPERIMENTAL_GENERATOR_POLICY` and `_MODEL_PRIORITY` from the same source (`targeted_generation.py`). They are structurally identical objects (tested by `test_18`). No model-asymmetry exists between conditions.
- **Scientific note**: Different fallback models are not interchangeable. Their identities remain visible in the ledger. Post-hoc model normalization is prohibited.

### Reference-Independence Threshold Calibration

- **Threshold**: `0.85` (frozen before main curriculum generation)
- **Method**: Normalized AST n-gram Jaccard token similarity on function-only canonical AST (`compute_ast_hashes` + `compute_structural_similarity` in `cfg_dedup.py`), applied to `solve` vs `solve_brute` only.
- **Calibration fixture**: `phase0_calibration/reference_fixture.py`
- **Fixture hash**: `707762ab31720b7c` (SHA-256 prefix of concatenated pair content)
- **Fixture result**: 11/11 assertions passed at threshold 0.85. All duplicate/near-duplicate classes score ≥ 0.85 (REJECT). All structurally distinct classes score < 0.06 (PASS). Threshold cleanly separates the two groups.
- **Fixture independence**: The fixture contains only synthetic engineering pairs constructed for threshold validation. It is independent of the MBPP calibration split, the HumanEval/MBPP-test evaluation sets, and any generated curriculum samples.
- **Freeze rule**: The threshold is frozen at `0.85` for the duration of Phase 0. Threshold adaptation based on accepted curriculum samples is prohibited. Any future change requires a new freeze revision.

### Generation Manifest

Before curriculum generation begins, `generation_manifest.py` writes `data/generation_manifest.json` (targeted) and `data/generation_manifest_uniform.json` (uniform). Each manifest answers:

| Question | Field |
|---|---|
| Which commit? | `code_commit` |
| Which freeze tag? | `freeze_tag` |
| Which generator model? | `generator_model` |
| Which fallback policy? | `fallback_policy` |
| Which temperature? | `generation_parameters.temperature` |
| Which RPM limit? | `rpm_limit` |
| Which batch size? | `batch_size` |
| Which quality-gate versions? | `quality_gate_versions` |
| Which dedup threshold? | `dedup_threshold` |
| Which calibration fixture? | `reference_calibration_fixture_hash` |
| Which reference-independence threshold? | `reference_independence_threshold` |
| Which seed policy? | `random_seed_policy` |

### V7 Scientific Repairs, Primary Eligibility & Target Attribution Specification

The Phase 0 V7 freeze addresses the causal attribution chain from base model observations to curriculum problems and establishes authoritative primary dataset gating:

1. **Authoritative Primary Eligibility Gate (`quality_gates.py`)**:
   - Single source of truth: `check_primary_targeted_eligibility()` and `check_primary_uniform_eligibility()`.
   - Distinguishes `PRIMARY_ELIGIBLE` from `AUXILIARY_NON_PRIMARY` and `REJECTED`.
   - **Targeted Primary Invariant**: Must be generated by primary model (`gemini-3.6-flash`) across both passes without fallback, verified by oracle, non-duplicate, subtype-valid, and empirically validated on $\theta_{\text{base}}$ (`target_validation_status == "VALIDATED"`).
   - **Uniform Primary Invariant**: Must be generated by primary model across both passes without fallback, verified by oracle, non-duplicate, and subtype-valid from the common support domain (Types A–F Monotonic Queue).

2. **Control Condition Scientific Symmetry**:
   - Both pipelines sample across the identical common algorithmic support (`SUBTYPES` Types A–F).
   - Uniform control receives no failure target information and does not require base-model target validation.
   - Preserves core causal claim: same support, same machinery, same budget, same generator policy; different targeting signal.

3. **Real Base-Model Probing Engine (`base_model_prober.py`)**:
   - Direct execution probing of $\theta_{\text{base}}$ (`Qwen/Qwen2.5-1.5B-Instruct`).
   - Measures greedy Pass@1 ($T=0.0$) and sampled Pass@10 ($k=10$, $T=0.8$, top-p=0.95).
   - Computes `empirical_difficulty = 1.0 - pass10`.
   - SHA-256 `probe_result_hash` cross-references probe records.

4. **Semantic Drift Prevention**:
   - Canonical `CLUSTER_SUBTYPE_COMPATIBILITY` mapping restricts subtype selection strictly to verified compatible subtypes for each cluster.
   - Incompatible targets or unrecognized clusters fail closed (`ValueError`).

5. **Generator Fallback Isolation**:
   - 2-pass tracking: `generator_model_pass1`, `generator_model_pass2`, `fallback_used_pass1`, `fallback_used_pass2`.
   - If fallback occurs in either pass, candidate is marked `AUXILIARY_NON_PRIMARY` and routed to auxiliary files (`data/v7_auxiliary_curriculum.jsonl`, `data/v7_uniform_auxiliary_curriculum.jsonl`). It never enters the primary dataset or counts toward the primary budget.

### Remaining Reproducibility Limitations
1. **API Non-Determinism**: Upstream hosted LLM sampling exhibits minor non-determinism even at fixed temperature and seed parameters.
2. **Fallback model isolation**: Fallback candidates are strictly quarantined to auxiliary datasets and segregated from primary experimental metrics.
3. **Process Resource Limits**: OS-level RLIMIT CPU/memory enforcement depends on POSIX `resource` availability and falls back gracefully under Windows environments.
4. **Pre-Experiment Freeze**: Quality gates, ledger schema, rate controls, and test suites are frozen at revision `phase0-experiment-freeze-v7`.
