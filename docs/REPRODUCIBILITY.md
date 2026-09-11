# VGCL Phase 0: Reproducibility Protocol & External Audit Specification

## I. Executive Summary & Archival State

This specification documents the technical architecture, execution protocols, and data contracts of the frozen **VGCL Phase 0** pipeline at commit `98398d6e49736d83fa3c39560d4d556a8a760419` (`98398d6`), tagged `phase0-experiment-freeze-v7` (11 September 2026).

The protocol was designed to evaluate the following research hypothesis under controlled experimental conditions:

> **"Failure-targeted verified synthetic curricula produce greater capability transfer than uniformly sampled verified synthetic curricula under identical compute budgets."**

**Status**: The experimental software pipeline and quality gates are fully implemented and verified (77/77 automated tests passing). However, the final empirical fine-tuning comparison was not executed. This document describes how the software functions for external audit and reconstruction.

---

## II. Final Implemented Protocol (V7)

The finalized Phase 0 architecture divides execution into three sequential phases to decouple resource-intensive GPU base-model probing from local API generation and verification.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. LOCAL WORKSTATION GENERATION & VERIFICATION                              │
│    python targeted_generation.py --mode local                               │
│    python uniform_generation.py --mode local                                │
│                                                                             │
│    - Generator: gemini-3.6-flash (EXPERIMENTAL_GENERATOR_POLICY)            │
│    - Pre-oracle syntax repair (max 1 component regeneration attempt)        │
│    - Reference independence check (threshold 0.85)                          │
│    - Subtype validity & cluster compatibility (Types A-F)                   │
│    - Two-phase differential oracle (Phase A: N<=8/5, Phase B: N>=100)       │
│    - Structural deduplication (AST/CFG similarity < 0.70)                   │
│    - Outputs candidate manifest: data/v7_probe_manifest.jsonl               │
│    - Guarded: Qwen weight downloading is blocked locally                    │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 2. REMOTE BASE-MODEL PROBING (Kaggle GPU)                                   │
│    python kaggle_probe.py --manifest data/v7_probe_manifest.jsonl           │
│                          --output data/v7_kaggle_probe_results.jsonl        │
│                                                                             │
│    - Base Model: Qwen/Qwen2.5-1.5B-Instruct                                 │
│    - Probing Policy: Greedy Pass@1, Sampled Pass@10 (k=10, T=0.8, top-p=0.95│
│    - Deterministic failure classification (FC_01 through FC_08)             │
│    - Outputs probe records with cryptographic SHA-256 provenance hashes     │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 3. LOCAL INGESTION & PRIMARY ELIGIBILITY GATING                             │
│    python merge_probe_results.py --probes data/v7_kaggle_probe_results.jsonl│
│                                                                             │
│    - Verifies cryptographic provenance and probing policy hashes            │
│    - Evaluates authoritative primary eligibility (quality_gates.py)         │
│    - Enforces target validation (target_validation_status == "VALIDATED")   │
│    - Isolates fallbacks to data/v7_auxiliary_curriculum.jsonl               │
│    - Writes verified primary items to data/v7_approved_curriculum.jsonl     │
│    - Updates immutable SQLite ledger: data/v7_experiment_ledger.db          │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## III. Protocol Components & Exact Parameters

### 1. Model Identities & Policies

| Role | Identifier | Configuration Parameters |
| :--- | :--- | :--- |
| **Primary Generator** | `gemini-3.6-flash` | Temperature: `0.4`, Pacing: `15 RPM`, Timeout: `30.0s` |
| **Fallback Generators** | `[gemini-3.1-flash-lite, gemini-3.5-flash-lite, gemini-flash-latest]` | Invoked only upon rate limit / quota exhaustion; isolated to auxiliary datasets |
| **Frozen Base Model** | `Qwen/Qwen2.5-1.5B-Instruct` | Greedy Pass@1 ($T=0.0$), Sampled Pass@10 ($k=10, T=0.8$, top-$p=0.95$, max tokens: $512$) |
| **Probing Policy Hash** | `_PROBING_POLICY_HASH` | SHA-256[:16] of canonical JSON decoding dictionary |

### 2. Local Hardware Guard (`base_model_prober.py`)

To prevent memory crashes and unexpected multi-gigabyte downloads on developer machines, `BaseModelProber` blocks weight loading unless:
- The environment variable `KAGGLE_KERNEL_RUN_TYPE` is set;
- The caller explicitly provides `allow_local_probe=True`; or
- The environment variable `VGCL_ALLOW_LOCAL_QWEN_PROBE=1` is exported.

If none of these conditions are met, the prober raises a `RuntimeError` describing the two-stage execution workflow.

### 3. Quality Gates & Verification Invariants

1. **Pre-Oracle Syntax Repair (`syntax_repair.py`)**:
   - Parses code with `ast.parse()`.
   - Normalizes markdown code fences and trailing whitespace.
   - If invalid, identifies the defective component (`solve_opt`, `solve_brute`, or `test_gen`) and executes at most one component-level regeneration attempt.
   - Problem spec, seeds, and valid components are immutable.

2. **Reference Independence Gate (`quality_gates.py`)**:
   - Extracts `solve` and `solve_brute`, unparses them to canonical AST representations, and measures normalized AST token n-gram Jaccard similarity.
   - Rejection threshold: $S \ge 0.85$ or identical CFG/AST shape.
   - Threshold calibrated against 11 pre-experiment test pairs in `phase0_calibration/reference_fixture.py` (fixture hash: `707762ab31720b7c`).

3. **Subtype Validity & Cluster Compatibility (`quality_gates.py`, `targeted_generation.py`)**:
   - Applies deterministic AST feature visitors (data structures, loop depths, recursion patterns) verifying compliance with Types A–F Monotonic Queue structures.
   - Enforces `CLUSTER_SUBTYPE_COMPATIBILITY`: Generation attempts for a target cluster are restricted strictly to compatible subtypes, preventing semantic drift.

4. **Two-Phase Differential Verification Oracle (`oracle_refactor.py`)**:
   - **Phase A (Correctness)**: Executes `solve` and `solve_brute` on identical small inputs. Scale cap is derived from brute solver AST complexity: pathological combinatorial/recursive solvers receive $N \le 5$; standard solvers receive $N \le 8$.
   - **Phase B (Constraint Scaling)**: Evaluates `solve` on large inputs ($N \ge 100$). Enforces time limits (2.0s), memory limits (512MB peak RSS via POSIX `getrusage` where available), and structural AST scaling checks (rejecting $O(N^2)$ nested loops on $N \ge 5000$).
   - *Verification Scope*: Confirms behavioral agreement under tested inputs; does not prove asymptotic complexity.

5. **Structural Deduplication (`cfg_dedup.py`)**:
   - Scope-aware AST identifier canonicalization, literal normalization, and Control Flow Graph (CFG) basic block/edge graph hashing.
   - Rejection threshold: $S_{\text{AST}} \ge 0.70$ against previously accepted corpus items.
   - *Deduplication Scope*: Reduces structural redundancy; does not prove semantic uniqueness.

6. **Authoritative Primary Eligibility Gate (`quality_gates.py`)**:
   - `check_primary_eligibility(entry, condition_type)` acts as the single source of truth:
     - Must be generated by `gemini-3.6-flash` across both passes without fallback.
     - Must satisfy verification oracle (Phase A pass count $>0$, Phase B pass, status `APPROVED`).
     - Must satisfy deduplication ($S < 0.70$).
     - Must have valid declared subtype within Types A–F common support.
     - For `TARGETED`: must possess verified base-model failure evidence (`target_validation_status == "VALIDATED"`) and matching cryptographic `probe_result_hash`.
     - For `UNIFORM`: must be sampled from common support without failure metadata (`target_validation_status == "NOT_APPLICABLE"`).
   - Items failing primary requirements due to fallback or unverified target status are routed to auxiliary datasets (`data/v7_auxiliary_curriculum.jsonl`) and excluded from primary curriculum budgets.

### 4. Evaluation Data Separation Invariant

Strict data separation is maintained to prevent benchmark contamination:
$$\text{Calibration} \cap \text{Training} \cap \text{Evaluation} = \emptyset$$

- **Calibration Split**: Exclusively `google-research-datasets/mbpp` (`train` split).
- **Evaluation Splits**: `HumanEval` and `MBPP` (`test` split).
- Legacy contaminated calibration files containing evaluation samples are archived in `data/archive/contaminated_calibration_v1/`.

---

## IV. Ledger Schema & Data Contracts

The SQLite ledger (`data/v7_experiment_ledger.db`) and JSONL streams maintain complete experimental provenance across 32 structured fields:

```sql
CREATE TABLE IF NOT EXISTS ledger (
    problem_id TEXT PRIMARY KEY,
    seed INTEGER,
    timestamp REAL,
    generator_model TEXT,
    generator_version TEXT,
    prompt_hash TEXT,
    target_failure_cluster TEXT,
    monotonic_queue_subtype TEXT,
    target_validation_status TEXT,
    target_evidence TEXT,
    probe_result_hash TEXT,
    generator_policy TEXT,
    generator_model_pass1 TEXT,
    generator_model_pass2 TEXT,
    fallback_used INTEGER,
    fallback_used_pass1 INTEGER,
    fallback_used_pass2 INTEGER,
    fallback_reason TEXT,
    fallback_reason_pass1 TEXT,
    fallback_reason_pass2 TEXT,
    primary_eligibility_status TEXT,
    primary_eligibility_reason TEXT,
    base_pass1 REAL,
    base_pass10 REAL,
    empirical_difficulty REAL,
    oracle_confidence REAL,
    phase_a_pass_count INTEGER,
    phase_b_pass INTEGER,
    verification_status TEXT,
    ast_hash TEXT,
    cfg_hash TEXT,
    similarity_score REAL,
    acceptance_reason TEXT,
    reviewer_override INTEGER,
    quarantine_reference TEXT,
    code_content TEXT,
    spec TEXT
);
```

---

## V. Pre-Registered Fine-Tuning Protocol (Unexecuted)

For future reference and external reproduction, the pre-registered LoRA training configuration intended for the empirical study is recorded below:

- **Target Architecture**: `Qwen/Qwen2.5-1.5B-Instruct`
- **LoRA Rank ($r$)**: `16`
- **LoRA Alpha ($\alpha$)**: `32`
- **Target Modules**: `["q_proj", "k_proj", "v_proj", "o_proj"]`
- **Learning Rate**: `2e-4`
- **Epochs**: `1`
- **Batch Size**: `8` (Micro-batch: 2, Gradient Accumulation: 4)
- **Optimizer**: `AdamW` ($\beta_1=0.9, \beta_2=0.999, \epsilon=1\times 10^{-8}$)
- **Weight Decay**: `0.01`
- **LR Scheduler**: `Cosine` with 5% warmup
- **Seeds**: `[42, 1337, 2026]`
- **Budget Matched Condition**: Equal total token and candidate counts between targeted and uniform cohorts.

---

## VI. Historical Prototypes vs Implemented Protocol

To prevent retrospective confusion, the table below explicitly contrasts superseded historical prototypes with the final frozen implementation:

| Dimension | Superseded Historical Behavior (V1–V5) | Final Frozen Implementation (V7) |
| :--- | :--- | :--- |
| **Failure Attribution** | Requested target label copied to candidate without verification | Real execution probing on `Qwen2.5-1.5B`; label marked `VALIDATED` only with runtime failure trace |
| **Difficulty Estimation** | Simulated via PRNG float generator (`random.uniform`) | Empirical difficulty calculated as $1.0 - \text{Pass@10}$ on base model |
| **Probing Execution** | Local Qwen execution attempted (causing OOM / disk exhaustion) | Two-stage pipeline: local generation + remote Kaggle GPU probing (`kaggle_probe.py`) |
| **Local Machine Guard** | None; attempted downloading full model weights | Local guard in `base_model_prober.py` blocks Qwen loading unless explicitly overridden |
| **Fallback Handling** | Fallback models silently accepted into curriculum | Fallback candidates strictly quarantined to `data/v7_auxiliary_curriculum.jsonl` |
| **Subtype Scheduling** | Unconstrained adaptive sampling drifted into easy subtypes | `CLUSTER_SUBTYPE_COMPATIBILITY` restricts sampling to valid cluster-subtype pairings |
| **Control Condition** | Unmatched or unstratified generic problems | Uniform control sampled from identical Types A–F Monotonic Queue domain without failure metadata |
| **Calibration Data** | Contaminated with HumanEval / MBPP test splits | Strictly restricted to MBPP `train` split ($\text{Calibration} \cap \text{Eval} = \emptyset$) |

---

## VII. External Audit Commands

Auditors can verify the integrity of the frozen repository using the following commands:

```bash
# 1. Verify git commit identity and tag
git rev-parse HEAD
# Output: 98398d6e49736d83fa3c39560d4d556a8a760419
git tag --points-at HEAD
# Output: phase0-experiment-freeze-v7

# 2. Verify clean compilation
python -m compileall -q -x "\.venv" .

# 3. Run full automated test suite (77 tests)
pytest test_phase0_pipeline.py

# 4. Verify reference independence calibration fixture (11 assertions)
python phase0_calibration/reference_fixture.py

# 5. Verify local Qwen guard raises RuntimeError locally
python -c "from base_model_prober import BaseModelProber; p = BaseModelProber(); p._ensure_model()"
```
