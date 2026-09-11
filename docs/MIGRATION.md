# VGCL Phase 0: Historical Architecture & Migration Record

## Executive Notice: Development Frozen

This document is a retrospective engineering record detailing the structural and architectural migrations executed across the development of the **VGCL Phase 0** pipeline up to its final archival freeze at commit `98398d6e49736d83fa3c39560d4d556a8a760419` (`98398d6`) on **11 September 2026**.

The migration procedures described below document how the pipeline evolved from early unconstrained prototypes into the final audited V7 architecture. They are preserved for historical provenance and reproducibility analysis, not as instructions for active development.

---

## 1. Summary of Major Systemic Migrations

Over the course of revisions V1 through V7, five major systemic migrations were executed to resolve scientific and engineering vulnerabilities:

```
[Early Prototype]
Paradigm-Driven Generation ──> Local Execution Attempt ──> Flat File Storage
                                      │
                                      ▼ (Audits Revealed Flaws)
                             - Memory crashes on local GPU
                             - PRNG fake difficulty
                             - Silent fallback contamination
                             - Evaluation data leakage
                             - Semantic drift in subtypes
                                      │
                                      ▼ (Architectural Repairs)
[Final V7 Frozen State]
Two-Stage Execution Split (Local API Generation + Remote Kaggle GPU Probing)
      │
      ├──> Local Hardware Guard (base_model_prober.py)
      ├──> Authoritative Primary Eligibility Gating (quality_gates.py)
      ├──> Strict Auxiliary Segregation of Fallback Outputs
      ├──> Cryptographic Provenance Hashing (SHA-256)
      └──> Dual SQLite + JSONL Immutable Ledger (experiment_ledger.py)
```

---

## 2. Detailed Architectural Transitions

### A. Transition from Paradigm-Driven to Failure-Targeted Generation
- **Historical State**: Early prototypes prompted the generator model for general competitive programming problems categorized under broad algorithmic themes (e.g., "monotonic queue problems").
- **Vulnerability**: Generating problems within an algorithmic theme provided no evidence that the problems addressed empirical execution weaknesses of the base model ($\theta_{\text{base}}$).
- **Migration**: Replaced thematic prompts with explicit failure cluster specifications (`FC_01` through `FC_08`), backed by execution-grounded failure classification (`failure_mapper.py`).

### B. Transition from Local Model Probing to the Two-Stage Local/Kaggle Architecture
- **Historical State**: Early implementations attempted to load `Qwen/Qwen2.5-1.5B-Instruct` directly in the local developer environment (Windows workstations with limited VRAM and system memory).
- **Vulnerability**: Local execution caused system out-of-memory errors, disk exhaustion from downloading model weights, and unreliable timeouts.
- **Migration**:
  1. Divided the pipeline into two decoupled execution stages:
     - **Stage 1 (Local Windows)**: API candidate generation, pre-oracle syntax repair, AST reference independence gating, subtype validation, two-phase differential verification, and CFG deduplication. Emits `data/v7_probe_manifest.jsonl`.
     - **Stage 2 (Remote Kaggle GPU)**: `kaggle_probe.py` loads `Qwen/Qwen2.5-1.5B-Instruct` in an environment with dedicated GPU resources, runs greedy Pass@1 and sampled Pass@10 execution probing, and exports structured results with cryptographic hashes to `data/v7_kaggle_probe_results.jsonl`.
  2. Implemented a strict **Local Hardware Guard** in `base_model_prober.py` that intercepts any local attempt to load Qwen weights and raises a descriptive `RuntimeError`.
  3. Implemented `merge_probe_results.py` to ingest remote probe outputs locally, verify cryptographic SHA-256 hashes against tampering, and perform primary eligibility gating.

### C. Transition from PRNG Difficulty Simulation to Real Execution Probing
- **Historical State**: Prior to V6, candidate difficulty was computed using pseudo-random number generator (PRNG) heuristics (`random.uniform`).
- **Vulnerability**: Simulated difficulty was scientifically meaningless and could not support empirical claims regarding curriculum difficulty or learning progression.
- **Migration**: Migrated to `BaseModelProber`, which measures real greedy Pass@1 and sampled Pass@10 ($k=10, T=0.8$) on `Qwen/Qwen2.5-1.5B-Instruct`. Empirical difficulty is formally defined as $1.0 - \text{Pass@10}$.

### D. Transition from Single Dataset Streams to Primary vs. Auxiliary Segregation
- **Historical State**: Earlier generation runs recorded all accepted candidates into a single curriculum file (`data/approved_curriculum.jsonl`), regardless of whether the generator model had fallen back to alternative models (`gemini-3.1-flash-lite`) during API rate-limit events.
- **Vulnerability**: Fallback models introduced unmonitored generator variance, violating the invariant of an identical, controlled generation policy.
- **Migration**:
  - Implemented decoupled two-pass tracking of generator models and fallback events (`generator_model_pass1`, `generator_model_pass2`, `fallback_used_pass1`, `fallback_used_pass2`).
  - Implemented `quality_gates.check_primary_eligibility`: candidates utilizing fallback are permanently routed to auxiliary datasets (`data/v7_auxiliary_curriculum.jsonl`) and prohibited from entering the primary dataset (`data/v7_approved_curriculum.jsonl`).

### E. Transition from Unconstrained Retries to Cluster-Subtype Compatibility
- **Historical State**: Subtype schedulers adaptively prioritized subtypes that had low acceptance counts, but without restricting which subtypes could be paired with which failure targets.
- **Vulnerability**: The generator drifted into producing easy-to-generate subtypes that were semantically incompatible with the requested failure cluster (e.g., requesting infinite-loop termination failures but generating sliding-window extrema problems).
- **Migration**: Enforced `CLUSTER_SUBTYPE_COMPATIBILITY` mapping inside `SubtypeScheduler.select()`, failing closed if a requested cluster has no valid compatible subtypes.

---

## 3. Historical Data Migration & Schema Backfill

When transitioning from the legacy flat-file format to the structured V7 SQLite ledger (`data/v7_experiment_ledger.db`), legacy records were imported using the following schema translation:

```python
import json
from cfg_dedup import compute_ast_hashes
from experiment_ledger import ExperimentLedger, LedgerEntry

ledger = ExperimentLedger(
    db_path="data/v7_experiment_ledger.db",
    jsonl_path="data/v7_approved_curriculum.jsonl",
    aux_jsonl_path="data/v7_auxiliary_curriculum.jsonl",
)

legacy_path = "data/archive/run_08_pre_phase0_approved_curriculum/approved_curriculum.jsonl"

with open(legacy_path, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        data = json.loads(line)
        code = data.get("code", "")
        ast_hash, cfg_hash, _ = compute_ast_hashes(code)

        # Legacy items imported with explicit UNVERIFIED attribution status
        entry = LedgerEntry(
            problem_id=f"LEGACY_{data.get('seed', 0)}",
            seed=data.get("seed", 0),
            target_failure_cluster="FC_01_MICRO_CORRECTNESS",
            monotonic_queue_subtype=data.get("paradigm", "Type A: sliding-window extrema"),
            target_validation_status="UNVERIFIED",
            target_evidence="{}",
            probe_result_hash="",
            generator_policy="EXPERIMENTAL_GENERATOR_POLICY",
            primary_eligibility_status="AUXILIARY_NON_PRIMARY",
            primary_eligibility_reason="Historical legacy record lacking empirical base-model probe trace",
            verification_status="APPROVED",
            ast_hash=ast_hash,
            cfg_hash=cfg_hash,
            code_content=code,
            spec=data.get("spec", ""),
        )
        ledger.record_entry(entry)
```

---

## 4. Final Frozen Execution Commands

To replicate the execution workflow of the final frozen V7 architecture:

```bash
# 1. Local Candidate Generation & Oracle Verification (Targeted)
python targeted_generation.py --target-failure-cluster FC_01_MICRO_CORRECTNESS --num-samples 50 --mode local

# 2. Local Candidate Generation & Oracle Verification (Uniform Control)
python uniform_generation.py --num-samples 50 --mode local

# 3. Remote Base-Model Probing (Executed on Kaggle GPU)
python kaggle_probe.py --manifest data/v7_probe_manifest.jsonl --output data/v7_kaggle_probe_results.jsonl

# 4. Local Ingestion & Primary Eligibility Merge
python merge_probe_results.py --probes data/v7_kaggle_probe_results.jsonl

# 5. Integration Test Verification (77 passing tests)
pytest test_phase0_pipeline.py
```
