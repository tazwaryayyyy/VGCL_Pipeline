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
