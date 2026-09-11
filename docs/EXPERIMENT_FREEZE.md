# VGCL Phase 0 Experiment Freeze

Freeze version:
phase0-experiment-freeze-v2

Previous freeze:
phase0-experiment-freeze-v1

Reason for revision:
Calibration leakage was discovered after v1 freeze.
HumanEval and MBPP test data were removed from calibration.

Current commit:
139ecd578cc17aa3c9ecead032d23c16a3071bc7

Date:
2026-08-08

Base model:
Qwen-2.5-1.5B-Instruct

Generator model:
Gemini-3.6-Flash

Calibration dataset:
MBPP train split only

Evaluation datasets:
HumanEval
MBPP test

Dataset separation:
Calibration ∩ Training ∩ Evaluation = empty

Training configuration:
LoRA rank:
16

Alpha:
32

Learning rate:
2e-4

Seeds:
42, 1337, 2026

Verification:
Two-phase differential oracle

Status:
Frozen for Phase 0 experiment execution

Throughput validation addendum:
Date:
2026-08-14

Scope:
Batch generation, pre-oracle syntax normalization, additive oracle rejection diagnostics, and pipeline metrics.

Validated semantics:
`--batch-size` dispatches independent candidate generation workers concurrently. It does not combine multiple candidates into one Gemini request. Verification, deduplication, calibration, and ledger recording are consumed in deterministic seed order.

Targeted scheduler:
Approval feedback is restricted to subtypes tied for the lowest accepted count, with planned batch slots reserved before dispatch. This preserves the failure-targeted accepted curriculum allocation while allowing adaptive attempt ordering within tied strata.

Syntax gate:
The pre-oracle syntax stage performs parse validation plus markdown-fence and trailing-whitespace normalization only. It is not a semantic code repair system.

Benchmark status:
No real Gemini throughput result is frozen here. Local mocked benchmarking validates batching mechanics only and must not be reported as API throughput.

Scientific Quality Gates addendum:
Date:
2026-08-15

Scope:
Four surgical scientific corrections:
1. Optimized/reference structural independence gate (threshold 0.85 near-isomorphism rejection).
2. Deterministic subtype validity gate (Types A–F structural AST checks, NOT_VALIDATABLE for unstratified identifiers).
3. Complexity-aware Phase A scaling (adaptive N<=5 bound for pathological brute solvers).
4. Controlled 1-shot component-level syntax failure regeneration with explicit retry accounting.

Status:
Verified and frozen with full test suite passing (30 tests).

---

# Freeze Revision: phase0-experiment-freeze-v3

Freeze version:
phase0-experiment-freeze-v3

Previous freeze:
phase0-experiment-freeze-v2

Reason for revision:
Final adversarial validation and freeze of scientific quality gates (reference independence, subtype validity, complexity-aware Phase A scaling, 1-shot syntax component regeneration, retry metrics accounting, and generator model policy).

Commit:
Validated state on main branch

Date:
2026-08-15

Base model:
Qwen-2.5-1.5B-Instruct

Generator model policy:
Primary: gemini-3.6-flash
Fallback priority: [gemini-3.6-flash, gemini-3.1-flash-lite, gemini-3.5-flash-lite, gemini-flash-latest]
Enforcement: Model identity recorded per-call in quarantine and ledger entries; identical priority order across targeted and uniform pipelines.

Calibration dataset:
MBPP train split only (disjoint from evaluation sets)

Evaluation datasets:
HumanEval, MBPP test

Quality Gates & Verification:
1. Pre-oracle syntax validation with 1-shot component-level LLM regeneration retry.
2. Optimized/reference structural independence gate (0.85 normalized AST n-gram similarity threshold, AST/CFG shape isomorphism check).
3. Subtype validity gate (deterministic AST evidence for Types A–F; NOT_VALIDATABLE for uniform/unstratified identifiers).
4. Two-phase differential verification oracle with complexity-aware Phase A scaling (N<=5 for recursive/combinatorial brute; N<=8 default) and Phase B (N>=100).
5. Structural corpus deduplication (AST/CFG similarity < 0.70).
6. Permanent quarantine protocol with structured rejection diagnostics.

Accounting & Metrics:
Separate accounting for initial generation API calls, syntax retry API calls, total API calls, syntax failures, syntax repairs, syntax retry successes, reference independence rejections, subtype rejections, Phase A/B passes and failures, deduplication rejections, and approvals.

Test Status:
30 integration tests passing (0 failures, 0 errors).
py_compile: passed.
git diff --check: passed.

Repository status:
Frozen for Phase 0 curriculum generation.

---

# Freeze Revision: phase0-experiment-freeze-v4

Freeze version:
phase0-experiment-freeze-v4

Previous freeze:
phase0-experiment-freeze-v3

Reason for revision:
Lock Phase 0 experimental generation configuration.
Introduces explicit auditable EXPERIMENTAL_GENERATOR_POLICY, pre-experiment threshold calibration fixture,
and frozen generation manifest. No scientific invariants changed.

Commit:
(see git log — commit "Lock Phase 0 experimental generation configuration")

Date:
2026-08-16

Base model:
Qwen-2.5-1.5B-Instruct

Generator model policy:
Primary: gemini-3.6-flash
Fallback sequence (free-tier quota failover): gemini-3.1-flash-lite → gemini-3.5-flash-lite → gemini-flash-latest
Policy declaration: EXPERIMENTAL_GENERATOR_POLICY in targeted_generation.py (imported identically by uniform_generation.py)
Enforcement: _MODEL_PRIORITY is constructed from EXPERIMENTAL_GENERATOR_POLICY; tested by test_16, test_17, test_18.
Fallback activation is logged with [FALLBACK-ROUTING]; model identity recorded in every ledger entry and quarantine artifact.
Different fallback models are not interchangeable; post-hoc normalization is prohibited.

Reference-independence threshold:
0.85

Threshold calibration:
Fixture file: phase0_calibration/reference_fixture.py
Fixture hash: 707762ab31720b7c
Fixture pairs: 11 total (Class 1-6 coverage)
Result: 11/11 assertions passed.
Duplicate/near-duplicate classes: min score 1.0000, max score 1.0000 — all REJECT at 0.85.
Distinct algorithm classes: max score 0.0507 — all PASS at 0.85.
Threshold cleanly separates both groups with no borderline ambiguity.
Fixture is independent of MBPP calibration, HumanEval, MBPP-test, and any curriculum samples.
Threshold is frozen. Adaptation based on curriculum samples is prohibited.

Generation configuration file:
phase0_calibration/config.yaml (extended with experimental_generator_policy and generation_configuration blocks)
Reference: generation_manifest.py produces data/generation_manifest.json at pipeline start.

Calibration dataset:
MBPP train split only (disjoint from evaluation sets)

Evaluation datasets:
HumanEval, MBPP test

Quality Gates & Verification:
1. Pre-oracle syntax validation with 1-shot component-level LLM regeneration retry.
2. Optimized/reference structural independence gate (0.85 normalized AST n-gram similarity threshold, AST/CFG shape isomorphism check).
3. Subtype validity gate (deterministic AST evidence for Types A-F; NOT_VALIDATABLE for uniform/unstratified identifiers).
4. Two-phase differential verification oracle with complexity-aware Phase A scaling (N<=5 for recursive/combinatorial brute; N<=8 default) and Phase B (N>=100).
5. Structural corpus deduplication (AST/CFG similarity < 0.70).
6. Permanent quarantine protocol with structured rejection diagnostics.

New in this revision:
- generation_manifest.py: auditable manifest written before every generation run.
- phase0_calibration/reference_fixture.py: deterministic pre-experiment threshold calibration fixture.
- phase0_calibration/config.yaml: extended with experimental_generator_policy and generation_configuration.
- EXPERIMENTAL_GENERATOR_POLICY: explicit named constant replacing implicit _MODEL_PRIORITY construction.
- Fallback routing: explicit [FALLBACK-ROUTING] log on every model switch; rate-limit detection broadened to include resource_exhausted and rate limit keywords.

Test Status:
37 integration tests passing (0 failures, 0 errors).
py_compile: passed for all 12 modules including generation_manifest.py.
git diff --check: passed.
Calibration fixture: 11/11 assertions passed.

Repository status:
Frozen for Phase 0 curriculum generation.

---

# Freeze Revision: phase0-experiment-freeze-v5

Freeze version:
phase0-experiment-freeze-v5

Previous freeze:
phase0-experiment-freeze-v4

Reason for revision:
Final pre-generation scientific hardening based on Gemini pilot observations:
1. Phase A Admissibility & Scale Cap Propagation (V5-A): Propagate explicit derived scale cap ($C$) to test generators, filter candidate test cases with exact boundary enforcement, and record structured scale diagnostics.
2. Phase B Constraint-Regime Multi-Scale Validation (V5-B): Extract problem constraint bounds ($N$, time limit, memory limit) and perform multi-scale test execution with AST structural scaling checks (rejecting $O(N^2)$ loops or unmemoized recursion on $N \ge 5000$, while permitting small quadratic domains $N \le 500$).
3. Generator Model Policy & Fallback Observability (V5-C): Classify API errors (`rate_limit`, `quota_exhaustion`, `transient_server_failure`, `request_rejection`, `unknown`) to isolate infrastructure failures from candidate quality denominators; log explicit fallback reasons; audit generator model serving distribution in `ExperimentLedger` and `PipelineMetrics`.
4. Empirical Difficulty Mathematical Audit (V5-D): Verified $1.0 - \text{base\_pass10}$ formula with explicit deterministic unit tests and edge-case boundary tests.

Commit:
(see git log — commit "Harden Phase 0 verification and generation control")

Date:
2026-08-23

Base model:
Qwen-2.5-1.5B-Instruct

Generator model policy:
Primary: gemini-3.6-flash
Fallback sequence: [gemini-3.1-flash-lite, gemini-3.5-flash-lite, gemini-flash-latest]
Policy declaration: EXPERIMENTAL_GENERATOR_POLICY in targeted_generation.py (imported identically by uniform_generation.py)
Observability: fallback reasons and classified error categories logged and recorded in metrics; serving model recorded in every ledger entry and quarantine artifact.

Quality Gates & Verification:
1. Pre-oracle syntax validation with 1-shot component-level LLM regeneration retry.
2. Optimized/reference structural independence gate (0.85 normalized AST n-gram similarity threshold, AST/CFG shape isomorphism check).
3. Subtype validity gate (deterministic AST evidence for Types A–F; NOT_VALIDATABLE for uniform/unstratified identifiers).
4. Two-phase differential verification oracle with explicit Phase A cap propagation ($C \le 5$ for recursive/combinatorial brute; $C \le 8$ default) and Phase B multi-scale constraint-regime validation.
5. Structural corpus deduplication (AST/CFG similarity < 0.70).
6. Permanent quarantine protocol with enriched rejection diagnostics.

Accounting & Metrics:
Separate accounting for API infrastructure events (`api_rate_limit_events`, `api_quota_events`, `api_transient_events`, `api_other_errors`, `api_fallback_events`), model serving counts, initial generation calls, syntax retry calls, reference independence rejections, subtype rejections, Phase A/B passes and failures, deduplication rejections, and approvals.

Test Status:
52 integration tests passing (0 failures, 0 errors).
py_compile: passed for all modules.
git diff --check: passed.
Calibration fixture: 11/11 assertions passed.

Repository status:
Frozen for Phase 0 curriculum generation.

---

# Freeze Revision: phase0-experiment-freeze-v6

Freeze version:
phase0-experiment-freeze-v6

Previous freeze:
phase0-experiment-freeze-v5

Reason for revision:
Final scientific repair of Phase 0 causal chain, base-model failure attribution, and generator policy enforcement:
1. Real Base-Model Probing Engine (`base_model_prober.py`): Replaced PRNG simulation (`calibrate_empirical_difficulty`) with deterministic execution probing of the frozen base model $\theta_{\text{base}}$ (`Qwen/Qwen2.5-1.5B-Instruct`), measuring greedy Pass@1 and sampled Pass@10 ($k=10$, $T=0.8$) under immutable probing policy hash.
2. Direct Failure Classification & Evidence Extraction: Probe execution failures are deterministically classified using `failure_mapper.classify_failure`, recording observable execution evidence (`exception_type`, `error_msg`, execution time, memory usage) into structured failure records.
3. Causal Target Attribution Ledger Schema (`experiment_ledger.py`): Ledger schema extended with `target_validation_status` (`VALIDATED` | `UNVERIFIED` | `NOT_APPLICABLE`), `target_evidence` (JSON trace), `probe_result_hash`, `generator_policy`, `fallback_used`, and `fallback_reason`. Invariant: `target_failure_cluster` records the requested target; `target_validation_status="VALIDATED"` certifies empirical failure evidence was observed on $\theta_{\text{base}}$.
4. Semantic Drift Prevention Policy (`targeted_generation.py`): Enforced canonical `CLUSTER_SUBTYPE_COMPATIBILITY` mapping inside `SubtypeScheduler.select()`. The eligible subtype stratum is strictly intersected with allowed subtypes for the requested failure cluster; unknown clusters or empty intersections fail closed with `ValueError`.
5. Generator Fallback Accounting & Anti-Contamination: `_call_llm_paced` tracks whether a fallback model was invoked. Fallback activations are recorded per candidate in `fallback_used` and `fallback_reason` in the ledger, preventing silent model promotion or unacknowledged generator contamination.

Date:
2026-08-30

Base model:
Qwen/Qwen2.5-1.5B-Instruct

Generator model policy:
Primary: gemini-3.6-flash
Fallback sequence: [gemini-3.1-flash-lite, gemini-3.5-flash-lite, gemini-flash-latest]
Policy declaration: EXPERIMENTAL_GENERATOR_POLICY

Quality Gates & Verification:
1. Pre-oracle syntax validation with 1-shot component-level LLM regeneration retry.
2. Optimized/reference structural independence gate (0.85 normalized AST n-gram similarity threshold, AST/CFG shape isomorphism check).
3. Subtype validity gate (deterministic AST evidence for Types A–F; NOT_VALIDATABLE for uniform/unstratified identifiers).
4. Subtype/cluster compatibility enforcement (fail-closed against semantic drift).
5. Two-phase differential verification oracle with explicit Phase A cap propagation ($C \le 5$ for recursive/combinatorial brute; $C \le 8$ default) and Phase B multi-scale constraint-regime validation.
6. Real base-model Pass@1/Pass@10 execution probing and failure attribution.
7. Structural corpus deduplication (AST/CFG similarity < 0.70).
8. Permanent quarantine protocol with enriched rejection diagnostics.

Test Status:
56 integration tests passing (0 failures, 0 errors).
py_compile: passed for all modules including `base_model_prober.py`.
git diff --check: passed.

# Freeze Revision: phase0-experiment-freeze-v7 (FINAL)

Freeze version:
phase0-experiment-freeze-v7

Previous freeze:
phase0-experiment-freeze-v6

Reason for revision:
Final pre-experiment scientific repair, primary dataset eligibility gating, and control symmetry alignment:
1. Authoritative Primary Eligibility Gate (`quality_gates.py`):
   - Implemented `check_primary_targeted_eligibility` and `check_primary_uniform_eligibility` as the single source of truth.
   - Distinguishes `PRIMARY_ELIGIBLE` from `AUXILIARY_NON_PRIMARY` and `REJECTED`.
   - Invariant: Only candidates produced by the frozen primary generator model (`gemini-3.6-flash`) across both decoupled passes without fallback, verified by oracle, non-duplicate, subtype-valid, and (for targeted) possessing confirmed empirical base-model failure attribution (`target_validation_status == "VALIDATED"`) are accepted into the primary curriculum.
2. Control Condition Scientific Symmetry (`uniform_generation.py`):
   - Aligned uniform control to sample from the identical common support domain (Types A–F Monotonic Queue / Deque structures) without receiving failure target cluster metadata.
   - Uniform control does NOT require base-model target failure validation, preserving true control condition causality: same support, same machinery, same budget, same generator policy; different targeting signal.
3. Generator Fallback Isolation & Anti-Contamination:
   - Decoupled two-pass model tracking (`generator_model_pass1`, `generator_model_pass2`, `fallback_used_pass1`, `fallback_used_pass2`, `fallback_reason_pass1`, `fallback_reason_pass2`).
   - If fallback occurs in either pass, candidate is immutably routed to auxiliary datasets (`data/v7_auxiliary_curriculum.jsonl`, `data/v7_uniform_auxiliary_curriculum.jsonl`) and does not count towards the primary curriculum sample budget.
4. Deterministic Behavioral Profiles (`CLUSTER_BEHAVIOR_PROMPTS`):
   - Added deterministic observable failure descriptions for all 8 clusters in `targeted_generation.py`.
5. Dataset Namespaces & Historical Immutability:
   - Historical V5/V6 data preserved. Primary experimental runs write exclusively to V7 namespaces (`v7_approved_curriculum.jsonl`, `v7_experiment_ledger.db`, `v7_uniform_approved_curriculum.jsonl`, `v7_uniform_experiment_ledger.db`).
6. Two-Stage Execution & Local Guard Architecture:
   - Preserves low-resource local Windows machines: Qwen/Qwen2.5-1.5B-Instruct weights and inference are strictly offloaded to remote Kaggle GPU execution via `kaggle_probe.py`.
   - Generation and oracle verification run locally with `--mode local`, registering candidates in the ledger as `PENDING_KAGGLE_PROBE` and emitting `data/v7_probe_manifest.jsonl`.
   - Remote probing records are cryptographically verified and merged into `data/v7_approved_curriculum.jsonl` via `merge_probe_results.py`.
   - Local guard ensures zero Qwen weight downloads on development machines without compromising the real base-model probe invariant.

Date:
2026-08-30

Base model:
Qwen/Qwen2.5-1.5B-Instruct

Generator model policy:
Primary: gemini-3.6-flash
Fallback sequence: [gemini-3.1-flash-lite, gemini-3.5-flash-lite, gemini-flash-latest]
Policy declaration: EXPERIMENTAL_GENERATOR_POLICY

Quality Gates & Verification:
1. Pre-oracle syntax validation with 1-shot component-level LLM regeneration retry.
2. Optimized/reference structural independence gate (0.85 normalized AST n-gram similarity threshold, AST/CFG shape isomorphism check).
3. Subtype validity gate (deterministic AST evidence for Types A–F).
4. Subtype/cluster compatibility enforcement (`CLUSTER_SUBTYPE_COMPATIBILITY`, fail-closed against semantic drift).
5. Two-phase differential verification oracle with explicit Phase A cap propagation ($C \le 5$ for recursive/combinatorial brute; $C \le 8$ default) and Phase B multi-scale constraint-regime validation.
6. Real base-model Pass@1/Pass@10 execution probing and failure attribution (`BaseModelProber`).
7. Structural corpus deduplication (AST/CFG similarity < 0.70).
8. Authoritative Primary Eligibility Gating (`check_primary_eligibility`).
9. Permanent quarantine and auxiliary routing protocol.

Test Status:
67 integration and unit tests passing (0 failures, 0 errors).
py_compile / compileall: passed for all modules.
git diff --check: passed.

Repository status:
LOCKED and FROZEN for primary Phase 0 experimental execution under `phase0-experiment-freeze-v7`.
