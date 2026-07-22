# VGCL Pipeline

VGCL Pipeline is a modular engine designed to generate, verify, and deduplicate competitive programming problems and code execution trajectories. It provides isolated verification, AST-based security checks, differential testing, and dynamic deduplication.

## Features

- **Robust LLM Parsing:** Fuzzy header extraction and handling for model output variations.
- **Sandboxed Execution:** AST validation and process isolation to block forbidden imports, system access, and IPC pipe deadlocks.
- **Differential Verification:** Cross-checks candidate solutions against brute-force implementations across scaled test inputs with float-safe comparison logic.
- **AST Deduplication:** Canonicalized structural indexing and adaptive n-gram sliding windows to filter redundant code topology.
- **Resilient Batch Processing:** Infrastructure rate-limit detection and exponential backoff to handle API quota limits without corrupting task status.

## Pipeline Architecture

```text
                   +-------------------+
                   |   LLM Generator   |
                   +---------+---------+
                             |
                             v
                   +-------------------+
                   |  Fuzzy Delimiter  |
                   |      Parser       |
                   +---------+---------+
                             |
                             v
                   +-------------------+
                   |   AST Security    |
                   |     Verifier      |
                   +---------+---------+
                             |
                             v
                   +-------------------+
                   | Differential Test |
                   |  Execution (IPC)  |
                   +---------+---------+
                             |
                             v
                   +-------------------+
                   |    AST Topology   |
                   |   Deduplication   |
                   +-------------------+
```

## Repository Structure

```text
├── phase0_calibration/     # Calibration data scripts and trajectory extraction
├── dedup.py                # AST canonicalization and Jaccard similarity engine
├── generator.py            # Prompts, API rate limit handlers, and response parser
├── runner.py               # Retry orchestrator and traceback sanitization
├── verifier.py             # IPC process isolation and differential execution
├── run_test_batch.py       # Production batch entrypoint across paradigms
└── test_production_engine.py # Test suite and smoke tests
```

## Quickstart

### Environment Setup

1. Clone the repository and navigate to the root directory:

```bash
git clone https://github.com/your-username/vgcl-pipeline.git
cd vgcl-pipeline
```

2. Set up a virtual environment and activate it:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Configure your API key in a `.env` file:

```env
GEMINI_API_KEY=your_api_key_here
```

### Execution

Run the smoke test protocol to verify process isolation and security checks:

```bash
python smoke_test.py
```

Run the full batch generation and verification pipeline:

```bash
python run_test_batch.py
```

## License
MIT
