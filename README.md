# VGCL Pipeline

A dataset generation pipeline for competitive programming problems. It prompts an LLM to write a problem spec, an optimized solution, a brute-force solution, and a test case generator, then verifies correctness before saving anything.

## How it works

1. **Generator** calls the Gemini API and parses the response into four blocks (spec, solve, brute, test generator).
2. **Verifier** runs the code in a sandboxed subprocess. It checks AST security, then differentially tests `solve` vs `solve_brute` on small-scale inputs. Large-scale inputs only run `solve` against a time and memory budget.
3. **Dedup engine** canonicalizes the AST and drops any candidate that is too similar to something already in the corpus.
4. Approved candidates are appended to `data/approved_curriculum.jsonl` immediately with `fsync`. If the run is interrupted and restarted, the pipeline reads that file on startup, pre-populates the dedup corpus, and resumes from the next unused seed.

## Files

```
generator.py          LLM calls, delimiter parser, exponential backoff
verifier.py           AST sandbox, differential execution, IPC worker processes
audit_curriculum.py   Generate audit curriculum and compare with approved curriculum
dedup.py              AST canonicalization, Jaccard n-gram dedup
runner.py             Retry loop with error pass-through to the LLM on retries
run_test_batch.py     Main batch entrypoint, resume logic, paradigm caps
smoke_test.py         Quick sanity check for IPC and security sandbox
test_production_engine.py  Full test suite
test_patch_targets.py      Unit tests for verifier and dedup patches
data/
  approved_curriculum.jsonl   Output dataset
  quarantine/                 Failed or duplicate candidates
```

## Setup

```bash
git clone https://github.com/your-username/vgcl-pipeline.git
cd vgcl-pipeline
python -m venv .venv
.venv\Scripts\activate   # or: source .venv/bin/activate
pip install -r requirements.txt
```

Add a `.env` file:

```
GEMINI_API_KEY=your_key_here
```

## Running

```bash
# Sanity check
python smoke_test.py

# Full test suite
python test_production_engine.py

# Audit existing curriculum
python audit_curriculum.py

# Batch generation (resumes automatically if data/approved_curriculum.jsonl exists)
python run_test_batch.py
```

Edit `items_per_paradigm` at the bottom of `run_test_batch.py` to control how many problems to generate per paradigm.

## License

MIT
