# VGCL Phase 0: Historical Architecture & Migration Record

## Executive Notice: Development Frozen

This document is a retrospective engineering record detailing the structural and architectural migrations executed across the development of the **VGCL Phase 0** pipeline through its V7 experimental freeze (`98398d6e49736d83fa3c39560d4d556a8a760419`, `11 September 2026`) and its V8 security-hardening freeze (`phase0-experiment-freeze-v8`, `19 November 2026`).

The migration procedures described below document how the pipeline evolved from early unconstrained prototypes into the audited V7 architecture and hardened V8 execution boundary. They are preserved for historical provenance and reproducibility analysis, not as instructions for active development.

---

## 1. Summary of Major Systemic Migrations

Over the course of revisions V1 through V8, six major systemic migrations were executed to resolve scientific, engineering, and security vulnerabilities:

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
                             - Untrusted in-process exec() / pickle IPC / trust_remote_code=True
                                      │
                                      ▼ (Architectural & Security Repairs)
[Final V7 + V8 Frozen State]
Two-Stage Execution Split (Local API Generation + Remote GPU Probing)
      │
      ├──> Local Hardware Guard & trust_remote_code=False (base_model_prober.py)
      ├──> Centralized Fail-Closed Sandbox Execution (isolated_executor.py)
      ├──> Authoritative Primary Eligibility Gating (quality_gates.py)
      ├──> Strict Auxiliary Segregation of Fallback Outputs
      ├──> Cryptographic Provenance Hashing (SHA-256)
      └──> Dual SQLite + JSONL Ledger (experiment_ledger.py)
```

---

## 2. Detailed Architectural Transitions

### A. Transition from Paradigm-Driven to Failure-Targeted Generation
- **Historical State**: Early prototypes prompted the generator model for general competitive programming problems categorized under broad algorithmic themes (e.g., "monotonic queue problems").
- **Vulnerability**: Generating problems within an algorithmic theme provided no evidence that the problems addressed empirical execution weaknesses of the base model ($\theta_{\text{base}}$).
- **Migration**: Replaced thematic prompts with explicit failure cluster specifications (`FC_01` through `FC_08`), backed by execution-grounded failure classification (`failure_mapper.py`).

### B. Transition from Local Model Probing to the Two-Stage Local/Kaggle Architecture
- **Historical State**: Early implementations attempted to load `Qwen/Qwen2.5-1.5B-Instruct` directly in the local developer environment (workstations with limited VRAM and system memory).
- **Vulnerability**: Local execution caused system out-of-memory errors, disk exhaustion from downloading model weights, and unreliable timeouts.
- **Migration**:
  1. Divided the pipeline into two decoupled execution stages:
     - **Stage 1 (Local)**: API candidate generation, pre-oracle syntax repair, AST reference independence gating, subtype validation, two-phase differential verification, and CFG deduplication. Emits `data/v7_probe_manifest.jsonl`.
     - **Stage 2 (Remote GPU Host)**: `kaggle_probe.py` loads `Qwen/Qwen2.5-1.5B-Instruct` on a compliant GPU host, runs greedy Pass@1 and sampled Pass@10 execution probing, and exports structured results with cryptographic hashes to `data/v7_kaggle_probe_results.jsonl`.
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

### F. Transition from In-Process `exec()` and Multiprocessing `pickle` to Fail-Closed Isolation (V8)
- **Historical State**: Through V7, `base_model_prober.py` evaluated combined candidate/reference/test-generator strings via in-process `exec()`, `legacy/audit_curriculum.py` used direct `exec()`, `oracle_refactor.py` and `legacy/verifier.py` used separate `multiprocessing` workers with `pickle` pipes and shared namespaces, and model loaders passed `trust_remote_code=True`.
- **Vulnerability**: Untrusted generated code executed inside the model-owning process or unhardened workers with access to secrets, host filesystem paths, network sockets, or forgeable IPC channels, while `trust_remote_code=True` allowed arbitrary remote model-repo code execution.
- **Migration**:
  1. Consolidated all untrusted execution (`base_model_prober.py`, `oracle_refactor.py`, `legacy/verifier.py`, `legacy/audit_curriculum.py`) into `isolated_executor.py`.
  2. Separated `generate_test_cases`, `solve`, and `solve_brute` into independent sandbox invocations with parent-side differential comparison over bounded lossless tagged JSON IPC (zero `pickle`).
  3. Implemented three-process non-dumpable supervisor/trusted-runner/candidate boundaries with per-run nonce authentication inside `ContainerSandbox` (digest-pinned Python image, `--network none`, `--read-only`, `--cap-drop ALL`, `--user 65534:65534`, zero host mounts) and `LinuxNamespaceSandbox` (unprivileged namespaces, `pivot_root`, `setrlimit`, architecture-verified `seccomp-bpf`).
  4. Added `kaggle_probe.py` fail-closed `get_active_sandbox()` preflight and enforced `trust_remote_code=False` across probing and calibration loaders.

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

To replicate the execution workflow of the final frozen V7 + V8 architecture (requires a compliant container runtime with the pinned Python digest image cached, or a Linux host passing `LinuxNamespaceSandbox` capability checks):

```bash
# 0. Pull pinned sandbox image if using container isolation
docker pull python@sha256:5e2bdb1b0d542d5039cffb8181c65930093949194a65c365dfb1b0fe13a4fa5b

# 1. Local Candidate Generation & Oracle Verification (Targeted)
python targeted_generation.py --target-failure-cluster FC_01_MICRO_CORRECTNESS --num-samples 50 --mode local

# 2. Local Candidate Generation & Oracle Verification (Uniform Control)
python uniform_generation.py --num-samples 50 --mode local

# 3. Remote Base-Model Probing (Executed on compliant GPU host)
python kaggle_probe.py --manifest data/v7_probe_manifest.jsonl --output data/v7_kaggle_probe_results.jsonl

# 4. Local Ingestion & Primary Eligibility Merge
python merge_probe_results.py --probes data/v7_kaggle_probe_results.jsonl

# 5. Full Integration & Security Test Verification (179 tests via pytest.ini)
VGCL_REQUIRE_SANDBOX=1 pytest -v
```
```

*(Note: verify `experiment_ledger.py` really has 37 columns as written in `CREATE TABLE IF NOT EXISTS ledger` — yes, the schema block itself in `REPRODUCIBILITY.md` has 37 columns, whereas the old text said "32 structured fields". Fixing "32" to "37" removes an internal mismatch in the doc!)*

---

### 3. Step-by-Step Git Commands (Clean Squash to `main`, Branch Deletion, `v8` Tag, Archive)

Run these commands inside `D:\desktopnew\RES\vgcl_pipeline`:

#### Step 1: Update the 4 `.md` files on `sec/sandbox-hardening-final` and verify CI

```powershell
git checkout sec/sandbox-hardening-final
git status
```

Save the updated contents above into `README.md`, `docs/EXPERIMENT_FREEZE.md`, `docs/REPRODUCIBILITY.md`, and `docs/MIGRATION.md`, then verify:

```powershell
git diff --check
python -m compileall -q -x "(\.venv|legacy/salvage_and_audit\.py|legacy/test_llm\.py)" .
pytest -v
```

Commit the doc updates on `sec/sandbox-hardening-final` and push so GitHub Actions confirms green on the branch before touching `main`:

```powershell
git add README.md docs/EXPERIMENT_FREEZE.md docs/REPRODUCIBILITY.md docs/MIGRATION.md
git commit -m "docs: update README and freeze docs for V8 security hardening"
git push origin sec/sandbox-hardening-final
```

Wait until **GitHub -> Actions** passes green on `sec/sandbox-hardening-final` (usually a couple of minutes; since `.md` changes sit alongside the workflow, let it finish green before squashing).

#### Step 2: Confirm `main` is still clean at `acbea49` and squash-merge once

```powershell
git fetch origin --tags --prune
git checkout main
git pull --ff-only origin main
git rev-parse HEAD
```

Confirm `git rev-parse HEAD` prints `acbea4949747a0966c2cc059418f960f5c506545`. Then squash-merge `sec/sandbox-hardening-final` onto `main`:

```powershell
git merge --squash sec/sandbox-hardening-final
git status
git diff --cached --stat
git diff --check --cached
```

Create the single clean commit on `main`:

```powershell
git commit -m "Harden candidate execution with isolated fail-closed sandboxes"
```

Push `main`:

```powershell
git push origin main
```

Wait for **GitHub -> Actions** on `main` to finish green so `main` itself has a passing badge/run recorded.

#### Step 3: Verify empty diff, delete `sec/sandbox-hardening-final`, and push `phase0-experiment-freeze-v8`

Confirm `main` and `sec/sandbox-hardening-final` have identical file contents (`git diff` prints nothing):

```powershell
git diff main sec/sandbox-hardening-final
```

Delete the remote and local branch (using `-D` because `--squash` does not link branch history):

```powershell
git push origin --delete sec/sandbox-hardening-final
git branch -D sec/sandbox-hardening-final
```

Grab the final `main` commit SHA (save this exact 40-character hash for Stage 2!) and create + push the V8 annotated tag:

```powershell
git rev-parse HEAD
git tag -a phase0-experiment-freeze-v8 -m "V8 freeze: fail-closed OS/container sandbox execution and documentation sync"
git push origin phase0-experiment-freeze-v8
```

#### Step 4: Create GitHub Release and Archive (Optional Final Lock)

1. On GitHub, go to **Releases -> Draft a new release**, select tag `phase0-experiment-freeze-v8`, title it `Phase 0 Security Hardening Freeze (V8)`, note that V7 research methodology/prompts/gating remain unchanged, execution is now fail-closed via `isolated_executor.py`, CI passed green, and no new GPU probing/training run was executed.
2. When completely done (and after grabbing `git rev-parse HEAD` for Stage 2), archive `tazwaryayyyy/VGCL_Pipeline` in **Settings -> General -> Danger Zone -> Archive this repository** if you want it read-only again.