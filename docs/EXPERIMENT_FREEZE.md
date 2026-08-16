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
