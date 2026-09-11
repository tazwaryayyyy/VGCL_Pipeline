# VGCL Pipeline

Verification-Guided Curriculum Learning (VGCL) is an experimental framework designed to test whether synthetic training curricula targeted at the empirically measured execution failure modes of a frozen base language model outperform uniformly sampled synthetic curricula under identical compute budgets. Candidates are generated via LLM passes, evaluated through an execution-grounded two-phase differential verification oracle, checked for structural novelty, and probed against a frozen base model to attribute observable execution failure evidence before curriculum ingestion.

## Research Question

The central hypothesis investigated by this project is:

> **"Failure-targeted verified synthetic curricula produce greater capability transfer than uniformly sampled verified synthetic curricula under identical compute budgets."**

This hypothesis specifies a controlled treatment-versus-control intervention. It is not an inquiry into general intelligence, universal curriculum superiority, or unconstrained synthetic data scaling.

## Final Status

**FROZEN / ARCHIVED — 11 September 2026**

The project reached a frozen experimental infrastructure state but was not taken through a final decisive treatment-vs-control training study. The repository therefore records a completed methodological and engineering investigation, not a confirmed empirical result. Neither a positive result, a negative result, nor a null result was conclusively established.

Development was intentionally terminated at this point rather than extended indefinitely. The remaining work would have required running and analyzing the final empirical treatment-vs-control training comparisons on downstream benchmarks. That work was not pursued further, so the repository is preserved as the frozen endpoint of the methodological investigation.

## Final Architecture (V7)

The finalized Phase 0 architecture separates generation, execution probing, and curriculum ingestion into a two-stage local and remote pipeline:

1. **Local Candidate Generation & Verification (Windows / Developer Environment)**:
   - Candidate problems, optimal solutions, and test generators are generated using `gemini-3.6-flash`.
   - Pre-oracle syntax repair performs whitespace/fence normalization and at most one isolated component-level syntax regeneration attempt.
   - Near-isomorphic reference/solution pairs are rejected by an AST n-gram structural independence gate ($S \ge 0.85$).
   - Deterministic structural checks validate declared algorithmic subtypes (Types A–F Monotonic Queue).
   - A two-phase differential verification oracle checks functional correctness against an independent brute-force reference on small scales ($N \le 5$ or $N \le 8$) and validates constraint scaling on large inputs ($N \ge 100$).
   - AST structural normalization and CFG hashing enforce corpus deduplication ($S < 0.70$).
   - Verified candidates are registered as pending probe in `v7_experiment_ledger.db` and exported to `data/v7_probe_manifest.jsonl`.
   - A local hardware guard prevents local downloading or memory allocation of `Qwen/Qwen2.5-1.5B-Instruct` under standard execution.

2. **Remote Base-Model Probing (Kaggle GPU Environment)**:
   - `kaggle_probe.py` ingests `data/v7_probe_manifest.jsonl` on GPU.
   - The frozen base model `Qwen/Qwen2.5-1.5B-Instruct` is probed under an immutable decoding policy (greedy Pass@1, sampled Pass@10 with $k=10, T=0.8$).
   - Real execution failures are deterministically classified into operational failure clusters (`FC_01` through `FC_08`).
   - Results, failure evidence traces, and SHA-256 provenance hashes are exported to `data/v7_kaggle_probe_results.jsonl`.

3. **Local Ingestion & Authoritative Primary Gating**:
   - `merge_probe_results.py` imports Kaggle probe records locally.
   - Cryptographic hashes, base-model identity, and probing policy hashes are verified against tampering.
   - `quality_gates.check_primary_eligibility` evaluates primary status: candidates with valid empirical failure evidence generated without fallback are admitted to `data/v7_approved_curriculum.jsonl`; fallback or unverified candidates are segregated into `data/v7_auxiliary_curriculum.jsonl`.

```
[Local Stage]
API Generation (Gemini-3.6-Flash)
      │
      ▼
Syntax Normalization & 1-Shot Component Retry
      │
      ▼
Reference Independence Gate (AST n-gram < 0.85)
      │
      ▼
Subtype Validity & Cluster Compatibility
      │
      ▼
Two-Phase Differential Oracle (Phase A Small + Phase B Large)
      │
      ▼
Structural Deduplication (AST Normalization + CFG < 0.70)
      │
      ▼
Export Candidate Manifest ──> data/v7_probe_manifest.jsonl
                                      │
[Remote Kaggle GPU Stage]             ▼
                      Base Model Probing (Qwen2.5-1.5B-Instruct)
                      Pass@1 (Greedy) & Pass@10 (k=10, T=0.8)
                      Deterministic Failure Attribution (FC_01..FC_08)
                      Cryptographic Provenance Hashing
                                      │
[Local Ingestion Stage]               ▼
Import & Verify Hashes <── data/v7_kaggle_probe_results.jsonl
      │
      ▼
Authoritative Primary Eligibility Gate
      ├──> Primary Eligible ──> data/v7_approved_curriculum.jsonl
      └──> Fallback / Auxiliary ──> data/v7_auxiliary_curriculum.jsonl
```

## Key Methodological Findings

Audits conducted across project revisions established several critical methodological insights:

1. **Paradigm-Driven Generation Was Insufficient**: Early prototypes prompted models for broad algorithmic themes. This did not guarantee that generated problems addressed actual failure modes of the target model.
2. **Requested Target $\neq$ Validated Target**: Simply instructing a generator to produce a problem for a specific failure cluster did not prove the resulting problem elicited that failure in the base model. True attribution required verifying execution failure traces directly on $\theta_{\text{base}}$.
3. **Synthetic PRNG Difficulty Confounded Measurement**: Early difficulty scores generated from PRNG simulation could not substantiate claims about empirical task hardness; they were replaced by real Pass@10 execution measurements.
4. **Subtype Retries Caused Semantic Drift**: Unconstrained adaptive scheduling allowed generation attempts to drift away from target failure clusters into easily generated, irrelevant subtypes. Enforcing a strict cluster-subtype compatibility matrix was necessary to prevent this drift.
5. **Fallback Generation Introduced Policy Contamination**: Free-tier API rate limiting caused silent fallbacks to alternative models (`gemini-3.1-flash-lite`), compromising generator comparability. Fallbacks had to be explicitly segregated into auxiliary non-primary datasets.
6. **Calibration Leakage Compromised Evaluation Integrity**: Early calibration sets inadvertently contained benchmark items, requiring an audited data separation protocol: $\text{Calibration} \cap \text{Training} \cap \text{Evaluation} = \emptyset$.

## Final V7 State

- **Commit**: `98398d6e49736d83fa3c39560d4d556a8a760419` (`98398d6`)
- **Freeze Tag**: `phase0-experiment-freeze-v7`
- **Automated Tests**: 77/77 passing (`pytest test_phase0_pipeline.py`)
- **Bytecode Compilation**: Clean (`python -m compileall -q -x "\.venv" .`)
- **Primary Generator**: `gemini-3.6-flash` (temperature 0.4, client-paced 15 RPM)
- **Base Model Probed**: `Qwen/Qwen2.5-1.5B-Instruct`
- **Verification Contracts**: Phase A ($N \le 5$ or $N \le 8$ small-scale differential equivalence) and Phase B ($N \ge 100$ scaling and resource checks).

Software tests confirm implementation integrity and adherence to declared interfaces. Passing 77/77 tests does not constitute scientific confirmation of the research hypothesis.

## Repository Documentation Guide

- [docs/EXPERIMENT_FREEZE.md](file:///d:/desktopnew/vgcl_pipeline/docs/EXPERIMENT_FREEZE.md): Authoritative freeze record documenting commit identity, quality gates, and full chronological revision history (V1 through V7).
- [docs/REPRODUCIBILITY.md](file:///d:/desktopnew/vgcl_pipeline/docs/REPRODUCIBILITY.md): Technical specification of execution stages, model configurations, local guard mechanisms, and verification invariants.
- [docs/MIGRATION.md](file:///d:/desktopnew/vgcl_pipeline/docs/MIGRATION.md): Historical record detailing architectural transitions from legacy unstratified generation to the two-stage V7 pipeline.

## Research Limitations

1. **Uncompleted Empirical Study**: No end-to-end model fine-tuning and evaluation experiment was completed. No effect size (positive, negative, or null) was established.
2. **Model Scale Scope**: All empirical probing was conducted on `Qwen/Qwen2.5-1.5B-Instruct`. No empirical conclusions can be drawn regarding larger models.
3. **Operational Failure Clusters**: The 8 failure clusters (`FC_01` through `FC_08`) are operational classifications based on observable runtime exceptions and timeouts, not proven mechanistic representations of neural reasoning.
4. **Oracle Verification Bounds**: Verification establishes behavioral agreement between optimized and brute-force implementations on tested inputs; it does not provide formal mathematical proofs of correctness or asymptotic complexity.
5. **Distributional Equivalence**: While targeted and uniform control pipelines share identical generator models, prompt formats, and algorithmic subtypes, subtle distributional differences between conditions can only be quantified in a fully completed empirical trial.

## Project Status

**Archival / Preserved.** No active development is underway. This repository is maintained strictly as an immutable record of the VGCL Phase 0 methodology and software infrastructure.
