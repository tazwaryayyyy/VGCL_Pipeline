# VGCL Pipeline

Verification-Guided Curriculum Learning (VGCL) is an experimental framework designed to test whether synthetic training curricula targeted at the empirically measured execution failure modes of a frozen base language model outperform uniformly sampled synthetic curricula under identical compute budgets. Candidates are generated via LLM passes, evaluated through an execution-grounded two-phase differential verification oracle, checked for structural novelty, and probed against a frozen base model to attribute observable execution failure evidence before curriculum ingestion.

## Research Question

The central hypothesis investigated by this project is:

> **"Failure-targeted verified synthetic curricula produce greater capability transfer than uniformly sampled verified synthetic curricula under identical compute budgets."**

This hypothesis specifies a controlled treatment-versus-control intervention. It is not an inquiry into general intelligence, universal curriculum superiority, or unconstrained synthetic data scaling.

## Final Status

**FROZEN / ARCHIVED - Experimental Freeze V7 (11 September 2026) & Security Freeze V8 (19 November 2026)**

The project reached a frozen experimental infrastructure state (V7) and subsequently received an execution-security freeze (V8) replacing in-process and ad-hoc worker execution with fail-closed OS/container sandboxing. The repository was not taken through a final decisive treatment-vs-control training study. It therefore records a completed methodological and engineering investigation, not a confirmed empirical result. Neither a positive result, a negative result, nor a null result was conclusively established.

Development was intentionally terminated rather than extended indefinitely. The remaining work would have required running and analyzing the final empirical treatment-vs-control training comparisons on downstream benchmarks. That work was not pursued further, so the repository is preserved as the frozen endpoint of the investigation.

## Final Architecture (V7 Methodology + V8 Execution Isolation)

The finalized Phase 0 architecture separates generation, execution probing, and curriculum ingestion into a two-stage local and remote pipeline, with all untrusted code execution isolated through `isolated_executor.py`:

1. **Local Candidate Generation & Verification**:
   - Candidate problems, optimal solutions, and test generators are generated using `gemini-3.6-flash`.
   - Pre-oracle syntax repair performs whitespace/fence normalization and at most one isolated component-level syntax regeneration attempt.
   - Near-isomorphic reference/solution pairs are rejected by an AST n-gram structural independence gate ($S \ge 0.85$).
   - Deterministic structural checks validate declared algorithmic subtypes (Types A-F Monotonic Queue).
   - A two-phase differential verification oracle (`oracle_refactor.py`) delegates to `isolated_executor.py` to check functional correctness against an independent brute-force reference on small scales ($N \le 5$ or $N \le 8$) and validate constraint scaling on large inputs ($N \ge 100$) inside fail-closed container or Linux namespace sandboxes.
   - AST structural normalization and CFG hashing enforce corpus deduplication ($S < 0.70$).
   - Verified candidates are registered as pending probe in `v7_experiment_ledger.db` and exported to `data/v7_probe_manifest.jsonl`.
   - A local hardware guard prevents accidental local downloading or memory allocation of `Qwen/Qwen2.5-1.5B-Instruct` under standard execution, and `trust_remote_code=False` is enforced across model/tokenizer loading.

2. **Remote Base-Model Probing**:
   - `kaggle_probe.py` checks `get_active_sandbox()` before loading `Qwen/Qwen2.5-1.5B-Instruct` and fails closed if neither a compliant container runtime nor Linux namespace + seccomp-bpf isolation passes verification.
   - When run on a compliant host, the frozen base model `Qwen/Qwen2.5-1.5B-Instruct` is probed under an immutable decoding policy (greedy Pass@1, sampled Pass@10 with $k=10, T=0.8$).
   - Candidate, reference, and test-generator code execute in separate sandbox invocations over bounded lossless tagged JSON IPC (no `pickle`), with differential comparison in the trusted host parent.
   - Real execution failures are deterministically classified into operational failure clusters (`FC_01` through `FC_08`), while sandbox infrastructure errors propagate as `SandboxInfrastructureError` rather than false candidate failures.
   - Results, failure evidence traces, and SHA-256 provenance hashes are exported to `data/v7_kaggle_probe_results.jsonl`.

3. **Local Ingestion & Authoritative Primary Gating**:
   - `merge_probe_results.py` imports probe records locally.
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
Two-Phase Differential Oracle (isolated_executor.py: Phase A Small + Phase B Large)
      │
      ▼
Structural Deduplication (AST Normalization + CFG < 0.70)
      │
      ▼
Export Candidate Manifest ──> data/v7_probe_manifest.jsonl
                                      │
[Remote Probing Stage]                ▼
                      Sandbox Preflight (get_active_sandbox, fail-closed)
                      Base Model Probing (Qwen2.5-1.5B-Instruct, trust_remote_code=False)
                      Pass@1 (Greedy) & Pass@10 (k=10, T=0.8) via isolated_executor.py
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
7. **Execution Isolation Required OS/Container Boundaries**: In-process `exec()`, shared execution namespaces, `pickle` IPC, and `trust_remote_code=True` exposed the model-owning parent and verification pipeline to untrusted code; V8 replaced them with centralized fail-closed sandboxes in `isolated_executor.py`.

## Final Frozen State (V7 + V8)

- **V7 Methodology Freeze**: Tag `phase0-experiment-freeze-v7` (`98398d6e49736d83fa3c39560d4d556a8a760419`, 11 September 2026), followed by docs commit `acbea49`.
- **V8 Security Hardening Freeze**: Tag `phase0-experiment-freeze-v8` (`main` `HEAD`, 19 November 2026).
- **Automated Tests**:
  - `test_phase0_pipeline.py`: 77/77 passing.
  - `test_security_sandbox.py`: 94 security and seccomp-BPF regression tests.
  - Full `pytest.ini` suite (`pytest -v`): 179 tests total, green on GitHub Actions (`.github/workflows/security-tests.yml` across Ubuntu 22.04 Python 3.10/3.11/3.12 with both `linux_ns` and `container` backends and `VGCL_REQUIRE_SANDBOX=1`, plus Windows fail-closed verification). On Windows with Docker Desktop locally, 170 pass and 9 Linux-only seccomp integration tests skip; on Linux with `linux_ns`, 179/179 pass.
- **Bytecode Compilation**: Clean (`python -m compileall -q -x "(\.venv|legacy/salvage_and_audit\.py|legacy/test_llm\.py)" .`).
- **Primary Generator**: `gemini-3.6-flash` (temperature 0.4, client-paced 15 RPM).
- **Base Model Probed**: `Qwen/Qwen2.5-1.5B-Instruct` (`trust_remote_code=False`).
- **Verification Contracts**: Phase A ($N \le 5$ or $N \le 8$ small-scale differential equivalence) and Phase B ($N \ge 100$ scaling and resource checks).

Software tests confirm implementation integrity, sandbox boundary controls, and adherence to declared interfaces. Passing the test suite does not constitute scientific confirmation of the research hypothesis.

## Repository Documentation Guide

- [SECURITY.md](SECURITY.md): Specification of the OS/container-level isolated candidate execution boundary, three-process trusted-runner architecture, seccomp-BPF policy, threat model, and residual risks.
- [docs/EXPERIMENT_FREEZE.md](docs/EXPERIMENT_FREEZE.md): Authoritative freeze record documenting commit identity, quality gates, and full chronological revision history (V1 through V8).
- [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md): Technical specification of execution stages, model configurations, sandbox requirements, and verification invariants.
- [docs/MIGRATION.md](docs/MIGRATION.md): Historical record detailing architectural transitions from legacy unstratified generation through V7 and V8 security isolation.

## Research Limitations

1. **Uncompleted Empirical Study**: No end-to-end model fine-tuning and evaluation experiment was completed. No effect size (positive, negative, or null) was established.
2. **Model Scale Scope**: All empirical probing was conducted on `Qwen/Qwen2.5-1.5B-Instruct`. No empirical conclusions can be drawn regarding larger models.
3. **Operational Failure Clusters**: The 8 failure clusters (`FC_01` through `FC_08`) are operational classifications based on observable runtime exceptions and timeouts, not proven mechanistic representations of neural reasoning.
4. **Oracle Verification Bounds**: Verification establishes behavioral agreement between optimized and brute-force implementations on tested inputs; it does not provide formal mathematical proofs of correctness or asymptotic complexity. Residual sandbox boundaries (shared kernel, user-namespace UID 0 mapping, OCI default seccomp vs custom BPF, and candidate-reported `per_case_times` under runner wall-clock caps) are documented in [SECURITY.md](SECURITY.md).
5. **Distributional Equivalence**: While targeted and uniform control pipelines share identical generator models, prompt formats, and algorithmic subtypes, subtle distributional differences between conditions can only be quantified in a fully completed empirical trial.

## Project Status

**Archival / Preserved.** No active development is underway. This repository is maintained strictly as an immutable record of the VGCL Phase 0 methodology and software infrastructure.
