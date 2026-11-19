# VGCL Security & Candidate Execution Architecture

This document specifies the security policy, execution boundaries, and sandboxing architecture for the VGCL Pipeline.

---

## 1. Threat Model

The VGCL Pipeline generates and evaluates synthetic Python code representing algorithmic solutions, brute-force references, and test generators. All model-generated and externally supplied code is treated as intentionally hostile.

### Untrusted Inputs

1. **Model-Generated Candidate Code (`solve_code`)**: Completions from `Qwen/Qwen2.5-1.5B-Instruct` or other models.
2. **Externally Supplied / LLM Reference Code (`solve_brute_code`)**: Sourced from manifests, LLM passes, or JSONL corpora.
3. **Externally Supplied / LLM Test Generators (`test_gen_code`)**: Code generating differential inputs.
4. **Oracle & Legacy Curriculum Code (`full_code`, `code_str`)**: Verified via `oracle_refactor.py`, `legacy/verifier.py`, and `legacy/audit_curriculum.py`.

### Protected Assets & Boundaries

- **Host Confidentiality**: Parent environment secrets (`GEMINI_API_KEY`, `KAGGLE_DATA_PROXY_TOKEN`, `HF_TOKEN`, SSH/cloud keys), parent process memory, `/proc/<ppid>/environ`, inherited file descriptors, and host files.
- **Host Integrity**: Repository source code, `.git/`, experiment ledgers (`data/v7_experiment_ledger.db`), and host temp directories.
- **Host Availability**: Protection against CPU starvation, memory exhaustion, fork/process bombs, disk/file flooding, and fd/stdout flooding.
- **Network Isolation**: OS/kernel-enforced denial of IPv4, IPv6, TCP, UDP, DNS, and host-loopback access.
- **Execution-Result Integrity**: Protection against candidate tampering with `solve_brute`, `generate_test_cases`, `are_results_equal`, or worker IPC verdicts.

---

## 2. Separate-Context & Three-Process Trusted-Runner Architecture

All candidate, oracle (`oracle_refactor.py`), and legacy (`legacy/verifier.py`, `legacy/audit_curriculum.py`) execution routes exclusively through `isolated_executor.py`:

1. **Separate Sandbox Contexts**: `generate_test_cases`, `solve`, and `solve_brute` execute in separate sandbox invocations. Neither `solve` nor `solve_brute` shares memory, globals, or runtime state with sibling untrusted functions.
2. **Three-Process Boundary Inside Each Sandbox**:
   - A non-dumpable **supervisor** (`PR_SET_DUMPABLE=0`) owns the host response channel. It generates a per-run random nonce, spawns the trusted runner, and accepts exactly one nonce-authenticated, bounded, single-frame response.
   - A non-dumpable **trusted runner** holds the supervisor response pipe write end (inherited via `pass_fds`, never exposed to candidate code). It spawns the candidate with `stdout=DEVNULL`, `stderr=DEVNULL`, and only a private data-pipe write end inherited. The runner independently enforces the wall-clock deadline, measures `execution_time` (candidate spawn→exit) and `memory_mb` (`getrusage(RUSAGE_CHILDREN).ru_maxrss`), enforces bounded output and single-frame framing, validates the data-only schema, and constructs the trusted envelope.
   - The **candidate** executes untrusted code. Its data frame may carry semantic results only (`status`, `outputs`, `cases`, `per_case_times`, `fail_idx`, diagnostics); any attempt to assert `execution_time`, `memory_mb`, or `passed` is rejected as `WorkerProtocolError`. A schema-valid forged `ok` frame written to an inherited descriptor therefore cannot become the trusted response or forge metrics.
3. **Trusted Parent Comparison**: Raw outputs are decoded losslessly in the trusted host parent, where `are_results_equal()` derives pass/fail verdicts.

---

## 3. Compliant Isolation Backends & Fail-Closed Policy

Backend selection is controlled by `VGCL_SANDBOX_BACKEND` (`auto`, `container`, `linux_ns`). Unknown or process-only values fail closed with `SandboxUnavailableError`. No process-only or in-process `exec()` fallback exists. Setting `VGCL_REQUIRE_SANDBOX=1` (used by CI) turns any backend-unavailable skip in the security test suite into a hard failure, so required infrastructure can never silently skip.

1. **OCI Container Backend (`ContainerSandbox` — Docker / Podman / optional `VGCL_CONTAINER_RUNTIME=runsc`)**:
   - Flags: `--rm --init --pull=never --network none --read-only --tmpfs /tmp:rw,noexec,nosuid,nodev,size=64m --cap-drop ALL --security-opt no-new-privileges --ipc none --user 65534:65534 --memory 512m --memory-swap 512m --cpus 1.0 --pids-limit 16 --ulimit fsize=1048576:1048576 --ulimit nofile=64:64 --ulimit cpu=... -w /tmp -i`.
   - Zero host bind mounts (`-v`). Image defaults to the digest-pinned `python@sha256:5e2bdb1b0d542d5039cffb8181c65930093949194a65c365dfb1b0fe13a4fa5b` (manifest digest of `python:3.12-slim`, verified via `docker pull` + `docker image inspect`). Mutable tags — including bare `python:3.12-slim` and any `:latest` — are rejected at construction time: the reference must end in `@sha256:<64 hex>`. The image must already exist in the local daemon cache (`image inspect` preflight); `--pull=never` forbids runtime downloads. Forces `rm -f` cleanup on timeout.
2. **Linux Namespace + Seccomp-BPF Backend (`LinuxNamespaceSandbox`)**:
   - Uses `unshare --user --map-root-user --net --pid --mount --fork` with `pivot_root` into an ephemeral minimal read-only rootfs (`/usr`, `/lib`, `/lib64`, `/bin`, plus read-only Python runtime prefix if external), fresh `/proc`, minimal `/dev/{null,zero,urandom}`, and private `noexec,nosuid,nodev` `tmpfs` `/tmp`.
   - Enforces `PR_SET_PDEATHSIG=SIGKILL`, `PR_SET_DUMPABLE=0`, `PR_SET_NO_NEW_PRIVS=1`, checked `setrlimit` (`RLIMIT_AS=512MB`, `RLIMIT_CPU`, `RLIMIT_FSIZE=1MB`, `RLIMIT_CORE=0`, `RLIMIT_NOFILE=64`, UID-relative `RLIMIT_NPROC`), and architecture-verified `seccomp-bpf` (`x86_64` / `aarch64`) blocking `socket`, `connect`, `bind`, `listen`, `accept`, `accept4`, `sendto`, `sendmsg`, `sendmmsg`, `ptrace`, `prctl`, `unshare`, `setns`, `mount`, `umount2`, `pivot_root`, `chroot`, `process_vm_readv`, `process_vm_writev`, `bpf`, `userfaultfd`, `io_uring_setup`, `keyctl`, and `clone(CLONE_NEW*)`.
   - Requires a live capability self-check at runtime before acceptance.

### Platform Constraints (Windows & Kaggle)

- **Windows**: Native Win32 Job Objects alone do not isolate filesystem reads or kernel networking and are rejected. Compliant execution on Windows requires `ContainerSandbox` (Docker Desktop / Podman with cached Linux Python image); otherwise execution fails closed with `SandboxUnavailableError`.
- **Kaggle Notebooks**: Standard Kaggle containers typically lack a Docker/Podman daemon and block unprivileged `unshare` namespaces. When neither compliant backend passes verification, `kaggle_probe.py` fails closed via `get_active_sandbox()` preflight before downloading or loading `Qwen/Qwen2.5-1.5B-Instruct` weights. Probing untrusted candidates requires a Linux VM/host with Docker/Podman or unprivileged namespaces + seccomp-bpf.

---

## 4. Safe IPC, Supply-Chain Hardening & Residual Risks

- **Lossless Tagged JSON IPC**: Zero `pickle` deserialization anywhere in execution paths. Supports `None`, `bool`, `int`, `float` (`nan`/`inf`/`-inf` tagged), lossless `str`, `bytes`, `list`, `tuple`, `set`, `frozenset`, `dict` (preserving non-string keys), and NumPy scalar unwrapping; rejects unsupported custom objects explicitly with `UnsupportedReturnTypeError`.
- **Bounded Framing**: Rejects non-finite JSON literals, trailing/multi-frame outputs, and unexpected keys; caps standard worker responses at `256 KB` and Phase B large test-gen inputs at `16 MB`. Infrastructure errors (`SandboxInfrastructureError`) propagate explicitly without mislabeling candidates as failed or quarantined.
- **Model Supply-Chain**: `trust_remote_code=False` is enforced across `base_model_prober.py`, `phase0_calibration/extract_trajectories.py`, and `phase0_calibration/prepare_calibration_data.py`.
- **Residual Risks**: Standard OCI containers and Linux namespaces share the host kernel (mitigable on Docker hosts using gVisor via `VGCL_CONTAINER_RUNTIME=runsc` or dedicated MicroVMs); kernel zero-days, hardware side channels, and compromised container daemons remain outside this boundary.
- **Metric Trust Scope**: `execution_time` and `memory_mb` are measured by the trusted runner, not the candidate. `per_case_times` remains candidate-reported (per-case timing would require per-case IPC); it is type-validated but not independently re-measured. Trusted enforcement covers the total wall-clock deadline, peak memory, and total execution time.
- **Backend Trust Scope**: The custom seccomp-BPF filter is enforced by the `linux_ns` backend; the container backend relies on the OCI runtime's default seccomp profile plus capabilities/rlimits/network isolation (its candidate runs as non-root `65534`, and the non-dumpable runner blocks `/proc` access to its descriptors). In `linux_ns`, the candidate is root inside the user namespace; the per-run nonce defeats blind writes to a discovered runner descriptor, but a root candidate could theoretically read runner memory via `/proc/<pid>/mem` (ptrace and `process_vm_readv/writev` syscalls are seccomp-blocked). This residual is inherent to `--map-root-user` namespace isolation.
