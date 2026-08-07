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