# VGCL Phase 0: Migration Architecture & Protocol Guide

This document outlines the procedure to migrate legacy synthetic dataset records into the **causally identifiable Phase 0 VGCL Experimental Framework**.

---

## 1. Architectural Overview of Phase 0 Changes

| Component | Legacy Pipeline | Phase 0 VGCL Pipeline |
| :--- | :--- | :--- |
| **Generator Drive** | Paradigm-driven iteration | **Failure-Cluster-Driven** (`failure_mapper.py` + `targeted_generation.py`) |
| **Control Group** | None | **Parallel Uniform Control** (`uniform_generation.py`) |
| **Verification Oracle** | Basic 4-tier check | **Two-Phase Differential Oracle** ($N \le 8$ small brute-force vs $N \ge 100$ large optimized) (`oracle_refactor.py`) |
| **Deduplication** | Token n-gram Jaccard | **AST Normalization + Control Flow Graph (CFG) Hashing** (`cfg_dedup.py`) |
| **Ledger Storage** | Flat JSONL | **Immutable SQLite + JSONL Experiment Ledger** (`experiment_ledger.py`) |
| **Quarantine Policy** | Ephemeral retry errors | **Permanent Trace & Artifact Quarantine** (`oracle_refactor.py`) |
| **Rate Control** | Ad-hoc backoff | **Deterministic Client-Side Pacing** (`sleep(60 / RPM_LIMIT)`) |

---

## 2. Legacy Dataset Ledger Backfill Migration

To migrate existing `data/approved_curriculum.jsonl` records into the new `experiment_ledger.db` structure:

```python
import json
from experiment_ledger import ExperimentLedger, LedgerEntry
from cfg_dedup import compute_ast_hashes

ledger = ExperimentLedger()
legacy_path = "data/approved_curriculum.jsonl"

with open(legacy_path, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        data = json.loads(line)
        code = data.get("code", "")
        ast_hash, cfg_hash, _ = compute_ast_hashes(code)
        
        entry = LedgerEntry(
            problem_id=f"LEGACY_{data.get('seed', 0)}",
            seed=data.get("seed", 0),
            target_failure_cluster="FC_01_MICRO_CORRECTNESS",
            monotonic_queue_subtype=data.get("paradigm", "Type A: sliding-window extrema"),
            verification_status="APPROVED",
            ast_hash=ast_hash,
            cfg_hash=cfg_hash,
            code_content=code,
            spec=data.get("spec", ""),
        )
        ledger.record_entry(entry)
print("Migration completed successfully.")
```

---

## 3. Running Phase 0 Experiments

### Step 1: Base Model Failure Mapping
```bash
python failure_mapper.py
```

### Step 2: Failure-Targeted Curriculum Generation
```bash
python targeted_generation.py --target-failure-cluster FC_01_MICRO_CORRECTNESS --num-samples 50
```

### Step 3: Uniform-Sampling Control Generation
```bash
python uniform_generation.py --num-samples 50
```

### Step 4: Capability Transfer Analysis
```bash
python transfer_matrix.py
```

### Step 5: Statistical Significance Testing
```bash
python statistical_analysis.py
```
