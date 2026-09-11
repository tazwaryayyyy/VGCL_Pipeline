# VGCL Phase 0: Authoritative Experiment Freeze Specification

## Archival Freeze Overview

| Metadata Field | Authoritative Frozen Value |
| :--- | :--- |
| **Freeze Tag** | `phase0-experiment-freeze-v7` |
| **Freeze Commit SHA** | `98398d6e49736d83fa3c39560d4d556a8a760419` (`98398d6`) |
| **Freeze Date** | **2026-09-11** |
| **Git Branch** | `main` (synced with `origin/main`) |
| **Development Status** | **FROZEN / ARCHIVED** |
| **Automated Test Suite** | **77/77 passing** (`pytest test_phase0_pipeline.py`) |
| **Bytecode Compilation** | **Clean** (`python -m compileall -q -x "\.venv" .`) |
| **Primary Generator Model** | `gemini-3.6-flash` (temperature `0.4`, `15 RPM` client pacing) |
| **Fallback Generator Models** | `[gemini-3.1-flash-lite, gemini-3.5-flash-lite, gemini-flash-latest]` (auxiliary-isolated) |
| **Frozen Base Model** | `Qwen/Qwen2.5-1.5B-Instruct` |
| **Base Model Probing Policy** | Greedy Pass@1 ($T=0.0$), Sampled Pass@10 ($k=10, T=0.8$, top-$p=0.95$) |
| **Probing Environment** | Remote Kaggle GPU (`kaggle_probe.py`) |
| **Local Machine Guard** | Active (`base_model_prober.py` blocks local model weight loading) |
| **Calibration Split** | `google-research-datasets/mbpp` (`train` split only) |
| **Evaluation Splits** | `HumanEval`, `MBPP` (`test` split) — strictly disjoint from calibration |
| **Dataset Separation** | $\text{Calibration} \cap \text{Training} \cap \text{Evaluation} = \emptyset$ |

---

## Why This Is an Archival Freeze

Development on the VGCL Pipeline was intentionally terminated on 11 September 2026. The project completed the construction, auditing, hardening, and verification of its experimental methodology and software infrastructure. Taking the project beyond this point would have required executing the downstream empirical fine-tuning and evaluation studies on compute clusters. That work was not pursued further.

The repository is frozen and archived at commit `98398d6` to serve as a permanent, unaltered reference of what was learned, what was implemented, what was verified through software testing, and what was left unresolved. There is no active roadmap and no planned subsequent phase.

---

## Final Scientific Status

The central research hypothesis formulated for this project is:

> **"Failure-targeted verified synthetic curricula produce greater capability transfer than uniformly sampled verified synthetic curricula under identical compute budgets."**

**The central treatment-vs-control hypothesis was NOT established by a completed final training and evaluation study.**

The repository reached a frozen experimental infrastructure state, but was not taken through a final decisive training comparison. Therefore:
- The repository records a completed methodological and architectural investigation, **not** a confirmed empirical result.
- No positive performance advantage for failure-targeted curriculum learning was established.
- No negative effect was established.
- No null result was conclusively demonstrated.
- Passing all 77 unit and integration tests certifies software correctness and adherence to declared protocol invariants; it does not constitute scientific evidence for the validity of the curriculum hypothesis.

---

## Architectural Specification (V7 State)

The final architecture addresses low-resource workstation constraints while maintaining empirical base-model probing rigor through a decoupled two-stage design:

```
+-------------------------------------------------------------------------------+
| LOCAL WORKSTATION STAGE (Windows)                                             |
|                                                                               |
|  1. Candidate Generation (Gemini-3.6-Flash, Decoupled 2-Pass)                  |
|  2. Pre-Oracle Syntax Repair & Whitespace Normalization (Max 1 retry)          |
|  3. Reference Independence Gate (AST n-gram similarity < 0.85)                 |
|  4. Subtype Validity Gate & Cluster Compatibility (Types A-F)                  |
|  5. Two-Phase Differential Oracle (Phase A Small N<=8, Phase B Large N>=100)   |
|  6. Structural Deduplication (Normalized AST + CFG graph hash, S < 0.70)       |
|  7. Ledger Staging (PENDING_KAGGLE_PROBE in SQLite ledger)                     |
|  8. Manifest Export (data/v7_probe_manifest.jsonl)                             |
|                                                                               |
|  [Local Guard]: Blocks Qwen/Qwen2.5-1.5B-Instruct weight download / loading    |
+-------------------------------------------------------------------------------+
                                      │
                                      ▼
+-------------------------------------------------------------------------------+
| REMOTE KAGGLE GPU STAGE                                                       |
|                                                                               |
|  1. Ingest Candidates from data/v7_probe_manifest.jsonl                        |
|  2. Load Frozen Qwen/Qwen2.5-1.5B-Instruct                                    |
|  3. Execute Greedy Pass@1 and Sampled Pass@10 (k=10, T=0.8, top-p=0.95)       |
|  4. Deterministic Failure Classification (FC_01 through FC_08)                |
|  5. Compute Cryptographic SHA-256 Provenance and Probing Policy Hashes         |
|  6. Export Probe Results to data/v7_kaggle_probe_results.jsonl                |
+-------------------------------------------------------------------------------+
                                      │
                                      ▼
+-------------------------------------------------------------------------------+
| LOCAL INGESTION & PRIMARY GATING STAGE                                        |
|                                                                               |
|  1. Ingest data/v7_kaggle_probe_results.jsonl via merge_probe_results.py        |
|  2. Cryptographically Verify Probing Policy and Result Hashes against Tampering|
|  3. Evaluate Primary Eligibility (quality_gates.check_primary_eligibility):   |
|       - Primary model used without fallback (both passes)                     |
|       - Oracle verified and structurally deduplicated                         |
|       - Subtype valid within declared common support                          |
|       - Targeted condition: target_validation_status == "VALIDATED"           |
|  4. Final Dataset Routing:                                                    |
|       - PRIMARY_ELIGIBLE    --> data/v7_approved_curriculum.jsonl              |
|       - AUXILIARY_NON_PRIMARY --> data/v7_auxiliary_curriculum.jsonl            |
+-------------------------------------------------------------------------------+
```

### Component Details

1. **Local Execution Guard (`base_model_prober.py`)**:
   - Blocks accidental downloading and initialization of `Qwen/Qwen2.5-1.5B-Instruct` on resource-constrained local Windows machines.
   - Raises `RuntimeError` unless running in a Kaggle environment (`KAGGLE_KERNEL_RUN_TYPE`), explicit local override (`allow_local_probe=True`), or environment variable `VGCL_ALLOW_LOCAL_QWEN_PROBE=1`.

2. **Primary Generator Policy (`EXPERIMENTAL_GENERATOR_POLICY`)**:
   - Declared identically across targeted and uniform pipelines:
     - Primary: `gemini-3.6-flash`
     - Fallbacks: `[gemini-3.1-flash-lite, gemini-3.5-flash-lite, gemini-flash-latest]`
     - Temperature: `0.4`, Client Timeout: `30.0s`, Pacing: `15 RPM`.
   - Any candidate involving fallback in Pass 1 or Pass 2 is marked `AUXILIARY_NON_PRIMARY` and written to `data/v7_auxiliary_curriculum.jsonl`. It never enters the primary curriculum or counts toward the experimental sample budget.

3. **Operational Failure Clusters (`failure_mapper.py`)**:
   - `FC_01_MICRO_CORRECTNESS`: Small-scale boundary or off-by-one errors.
   - `FC_02_SCALE_DIVERGENCE`: Passes micro-scale, fails large-scale time complexity.
   - `FC_03_TERMINATION`: Infinite recursion, iteration cycles, or timeouts.
   - `FC_04_BOUNDS_VIOLATION`: Container out-of-bounds (`IndexError`, `KeyError`).
   - `FC_05_RESOURCE_FAILURE`: Excessive auxiliary space or peak memory RSS (>512MB).
   - `FC_06_FORMAT_CONTRACT`: Return type mismatch or function signature violation.
   - `FC_07_STATE_INCONSISTENCY`: In-place input argument mutation or cross-query state leakage.
   - `FC_08_UNKNOWN`: General unclassified deterministic execution mismatch.
   - *Status*: Operational and observable runtime categories, not proven mechanistic representations of neural reasoning.

4. **Reference Independence Gate (`quality_gates.py`)**:
   - Evaluates function-only AST n-gram token similarity between `solve` and `solve_brute`.
   - Threshold: `0.85` (calibrated against 11 pre-experiment test pairs in `phase0_calibration/reference_fixture.py`, fixture hash `707762ab31720b7c`).
   - Rejects near-isomorphic and identical CFG/AST implementations before oracle execution.

5. **Two-Phase Differential Oracle (`oracle_refactor.py`)**:
   - Phase A: Evaluates functional equivalence between `solve` and `solve_brute` on small inputs. Complexity-aware scaling limits pathological brute-force solvers (recursion, combinatorial calls) to $N \le 5$, with standard solvers capped at $N \le 8$.
   - Phase B: Evaluates `solve` on large scale inputs ($N \ge 100$), multi-scale constraint compatibility, memory RSS, and structural loop nesting.
   - *Status*: Verifies behavioral compliance under tested oracle inputs; does not constitute a formal mathematical proof of correctness or complexity bounds.

6. **Structural Deduplication (`cfg_dedup.py`)**:
   - AST scope-aware variable/literal normalization + Control Flow Graph (CFG) basic-block and edge extraction.
   - Rejects candidates with structural similarity $S \ge 0.70$ against previously accepted corpus items.
   - *Status*: Reduces structural redundancy across the corpus; does not prove semantic uniqueness.

7. **Target Attribution Invariant**:
   - A requested target cluster specified during generation (`target_failure_cluster`) does not confer primary eligibility.
   - Only observed, classified execution failures on `Qwen/Qwen2.5-1.5B-Instruct` recorded in `target_evidence` and marked `target_validation_status = "VALIDATED"` satisfy the targeted primary gate.

8. **Control Condition Symmetry (`uniform_generation.py`)**:
   - Shares identical algorithmic domain support (Types A–F Monotonic Queue / Deque structures).
   - Uses identical generator model, prompt format contracts, quality gates, and verification oracle.
   - Does not receive target failure cluster guidance and does not require base-model target validation, preserving true treatment-control contrast: identical machinery and compute budget, different targeting signal.

---

## Chronological Evolution: Revisions V1 through V7

The safeguards and architecture documented above did not exist in the initial prototype. They were developed sequentially in response to methodological weaknesses identified during audits.

### Phase 0 Prototype (Pre-Freeze)
- Initial prototype generated synthetic algorithmic problems guided purely by high-level paradigm themes.
- Lacked base-model probing, empirical difficulty calibration, structural deduplication, and formal verification gates.
- Did not constitute an experimental test of failure targeting.

### Revision V1 (`phase0-experiment-freeze-v1` — Commit `d0a6570`)
- **Focus**: Initial formal pipeline freeze.
- **Flaw Discovered**: Calibration dataset included items from HumanEval and MBPP test splits, introducing severe data contamination risk into downstream evaluation.

### Revision V2 (`phase0-experiment-freeze-v2` — Commit `674a66d`)
- **Focus**: Evaluation data leakage repair.
- **Repair**: HumanEval and MBPP test splits were removed from calibration. Calibration was restricted exclusively to `google-research-datasets/mbpp` (`train` split).
- **Invariant**: Established $\text{Calibration} \cap \text{Training} \cap \text{Evaluation} = \emptyset$. Legacy contaminated artifacts archived.

### Revision V3 (`phase0-experiment-freeze-v3` — Commit `4e29e0a`)
- **Focus**: Scientific quality gates implementation.
- **Flaws Discovered**:
  - LLM frequently generated identical or trivial variations of `solve` for `solve_brute`, undermining differential oracle validity.
  - Generated code frequently diverged from declared algorithmic subtypes.
  - Pathological recursive brute-force references timed out during Phase A verification.
- **Repairs**:
  - Introduced AST n-gram reference independence gate (0.85 threshold).
  - Introduced deterministic AST subtype validity gate for Types A–F.
  - Added complexity-aware Phase A scale derivation ($N \le 5$ for combinatorial brute solvers).
  - Added controlled 1-shot component-level syntax repair.

### Revision V4 (`phase0-experiment-freeze-v4` — Commit `c4b2411`)
- **Focus**: Experimental configuration locking.
- **Flaws Discovered**: Implicit fallback routing allowed different generator models to serve requests without audit tracking.
- **Repairs**:
  - Introduced explicit `EXPERIMENTAL_GENERATOR_POLICY`.
  - Added pre-experiment reference independence threshold calibration fixture (`phase0_calibration/reference_fixture.py`, hash `707762ab31720b7c`).
  - Added auditable `generation_manifest.py`.

### Revision V5 (`phase0-experiment-freeze-v5` — Commit `5ff6c3f`)
- **Focus**: Verification hardening and scale propagation.
- **Flaws Discovered**:
  - Test generators produced test cases exceeding Phase A scale policies.
  - Solutions with $O(N^2)$ complexity passed Phase B on large regimes.
  - Difficulty scores were still simulated via PRNG heuristics.
- **Repairs**:
  - Exact scale cap propagation and boundary enforcement in Phase A.
  - Multi-scale constraint regime validation in Phase B, rejecting nested loops on $N \ge 5000$.
  - Structured API error classification.
  - Verified empirical difficulty mathematical formulation ($1.0 - \text{pass10}$).

### Revision V6 (`phase0-experiment-freeze-v6` — Commit `98c9969`)
- **Focus**: Real base-model probing and target attribution.
- **Flaws Discovered**:
  - PRNG difficulty simulation could not support empirical difficulty claims.
  - Copying requested failure labels onto curriculum items lacked empirical basis.
  - Subtype schedulers exhibited semantic drift away from target failure clusters.
- **Repairs**:
  - Replaced PRNG difficulty with real execution probing of `Qwen/Qwen2.5-1.5B-Instruct` (`base_model_prober.py`).
  - Added deterministic failure classification and evidence extraction (`failure_mapper.py`).
  - Extended ledger schema with `target_validation_status` (`VALIDATED` vs `UNVERIFIED`) and `target_evidence`.
  - Enforced `CLUSTER_SUBTYPE_COMPATIBILITY` mapping to eliminate semantic drift.

### Revision V7 (`phase0-experiment-freeze-v7` — Commit `98398d6`)
- **Focus**: Two-stage remote execution, local guard, and authoritative primary gating.
- **Flaws Discovered**:
  - Attempting to download and run Qwen-2.5-1.5B locally exhausted memory and disk on development machines.
  - Generator fallback under rate limits contaminated primary datasets with non-primary model samples.
  - Uniform control required structural alignment with the targeted condition's algorithmic support domain.
- **Repairs**:
  - Architected two-stage pipeline: local generation/verification and remote Kaggle GPU probing (`kaggle_probe.py`).
  - Implemented local hardware guard in `base_model_prober.py` to block accidental local Qwen weight downloads.
  - Implemented authoritative primary eligibility gate in `quality_gates.py`, quarantining all fallback outputs to auxiliary datasets.
  - Aligned uniform control to sample across identical Monotonic Queue subtypes (Types A–F) without failure targeting metadata.
  - Full test suite verified at 77/77 passing tests.

---

## Tag and Commit Integrity Record

- **Authoritative Identity**: The authoritative identity of this final freeze is commit SHA `98398d6e49736d83fa3c39560d4d556a8a760419`.
- **Tag State**: Tag `phase0-experiment-freeze-v7` points to commit `98398d6`.
- **Note on Git Tags**: Because git tags can be mutable across remotes, external auditors must treat the commit SHA (`98398d6`) as the immutable ground truth for the final frozen state.

---

## Summary of Unanswered Research Questions

1. Does failure-targeted synthetic curriculum learning produce statistically significant improvements in capability transfer compared to uniform curriculum learning under matched compute budgets?
2. What is the empirical transfer matrix $M_{k,j}$ across failure clusters when fine-tuning a 1.5B model?
3. How do findings from 1.5B base models transfer to larger dense or mixture-of-experts architectures?
4. Can operational failure clusters based on runtime exceptions accurately approximate the underlying latent failure distributions of language models?
