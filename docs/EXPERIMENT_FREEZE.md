# VGCL Phase 0 Experiment Freeze

## Freeze Metadata

Freeze version:
phase0-experiment-freeze-v1

Freeze commit:
312a6c5d9ce00d208d7f412e642ea33bcc5a8253

Freeze date:
2026-08-07

Repository:
VGCL_Pipeline


## Models

Base model:
Qwen-2.5-1.5B-Instruct

Generator model:
Gemini-3.6-Flash


## Pipeline

Experiment:
Verification-Guided Curriculum Learning (VGCL) Phase 0


## Experimental Conditions

Target condition:

Failure-targeted verified synthetic curriculum generation.


Control condition:

Uniform verified synthetic curriculum generation.


## Verification

Oracle:

Two-phase differential verification.

Phase A:
Small-input brute-force correctness verification.

Phase B:
Large-input optimized execution constraint verification.


## Training Configuration

Method:
LoRA fine-tuning

Rank:
16

Alpha:
32

Learning rate:
2e-4

Seeds:
42, 1337, 2026


## Evaluation

Primary metric:

Cluster-specific Pass@1 improvement.


Secondary metrics:

- Transfer matrix M(k,j)
- Overall Pass@1
- Pass@10
- Cross-cluster transfer


## Status

Repository frozen for Phase 0 experimental execution.

Dataset generation and training have not started yet.