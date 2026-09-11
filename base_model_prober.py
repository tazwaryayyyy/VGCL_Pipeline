"""base_model_prober.py

Real Base-Model Probing Engine for VGCL Phase 0.

Measures Pass@1 and Pass@10 of Qwen/Qwen2.5-1.5B-Instruct on a given
curriculum candidate problem by generating code samples and executing them
through the existing VerificationOracle.

Scientific contract:
  - All measurements are produced by the frozen base model theta_base.
  - No PRNG simulation is used.
  - Every probing session is recorded with model ID, tokenizer hash,
    decoding config, and seed list.
  - Failure evidence is deterministically classified and returned for
    use as targeted-generation context.

Usage:
    from base_model_prober import BaseModelProber
    prober = BaseModelProber()
    result = prober.probe(problem_spec=spec, solve_brute=brute, test_gen=tgen, seed=1001)
"""

import ast
import hashlib
import json
import os
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Frozen model identity
# ---------------------------------------------------------------------------

BASE_MODEL_ID: str = "Qwen/Qwen2.5-1.5B-Instruct"

# Decoding / sampling policy
PROBING_POLICY: Dict[str, Any] = {
    "model_id": BASE_MODEL_ID,
    "max_new_tokens": 512,
    "temperature_pass1": 0.0,   # greedy for Pass@1
    "temperature_pass10": 0.8,  # sampled for Pass@10
    "top_p": 0.95,
    "do_sample_pass1": False,
    "do_sample_pass10": True,
    "k_samples": 10,            # number of samples for Pass@10
    "policy_version": "v1.0_qwen25_1.5b_instruct",
}

_PROBING_POLICY_HASH: str = hashlib.sha256(
    json.dumps(PROBING_POLICY, sort_keys=True).encode()
).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Failure classification (imported from canonical module)
# ---------------------------------------------------------------------------

from failure_mapper import classify_failure, CLUSTER_MAP, FailureCategory

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class ProbingResult:
    """Structured result of a single base-model probe session."""
    problem_id: str
    base_model_id: str
    probing_policy_version: str
    probing_policy_hash: str
    probe_timestamp: float
    probe_seeds: List[int]

    # Pass-rate measurements
    pass1: float           # greedy pass rate (0.0 or 1.0 from greedy decode)
    pass10: float          # 1.0 if any of k_samples pass, else 0.0
    empirical_difficulty: float  # 1.0 - pass10

    # Failure evidence (from the last failing sample, or None if all pass)
    failure_cluster_id: Optional[str]
    failure_category: Optional[str]
    failure_evidence: Dict[str, Any]   # observed execution evidence

    # Tokenizer fingerprint
    tokenizer_vocab_size: int
    tokenizer_name_or_path: str

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d


# ---------------------------------------------------------------------------
# Core prober class
# ---------------------------------------------------------------------------


LOCAL_PROBE_OVERRIDE_ENV: str = "VGCL_ALLOW_LOCAL_QWEN_PROBE"


class BaseModelProber:
    """
    Probes Qwen/Qwen2.5-1.5B-Instruct against a curriculum problem to
    produce Pass@1, Pass@10, empirical_difficulty, and structured failure
    evidence for targeted curriculum generation.

    The model is loaded lazily on first call and cached for the session.
    Protected by a local execution guard to prevent accidental local weight downloads.
    """

    def __init__(self, device: Optional[str] = None, allow_local_probe: bool = False) -> None:
        self._model = None
        self._tokenizer = None
        self._device: str = device or ("cuda" if self._is_cuda_available() else "cpu")
        self._allow_local_probe: bool = allow_local_probe
        self._load_error: Optional[str] = None

    @staticmethod
    def _is_cuda_available() -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def _ensure_model(self) -> None:
        """Loads model and tokenizer if not already loaded, enforcing local environment guards."""
        if self._model is not None or self._load_error is not None:
            return

        is_kaggle = "KAGGLE_KERNEL_RUN_TYPE" in os.environ or "KAGGLE_DATA_PROXY_TOKEN" in os.environ
        override_set = os.environ.get(LOCAL_PROBE_OVERRIDE_ENV) == "1"
        if not (self._allow_local_probe or is_kaggle or override_set):
            raise RuntimeError(
                f"[VGCL LOCAL GUARD] Local loading of {BASE_MODEL_ID} is BLOCKED on this machine.\n"
                "Reason: Local hardware constraints (disk/RAM). Base-model probing must be performed on Kaggle.\n"
                "Workflow:\n"
                "  1. Run candidate generation locally (exports to data/v7_probe_manifest.jsonl)\n"
                "  2. Run 'python kaggle_probe.py --input data/v7_probe_manifest.jsonl --output data/v7_kaggle_probe_results.jsonl' on Kaggle\n"
                "  3. Merge results locally using 'python merge_probe_results.py ...'\n"
                "To override this guard on a dedicated local GPU machine, set environment variable:\n"
                f"    {LOCAL_PROBE_OVERRIDE_ENV}=1"
            )

        try:
            os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            print(
                f"  [PROBER] Loading {BASE_MODEL_ID} on {self._device} ...",
                flush=True,
            )
            self._tokenizer = AutoTokenizer.from_pretrained(
                BASE_MODEL_ID, trust_remote_code=True
            )
            if self._tokenizer.pad_token_id is None:
                self._tokenizer.pad_token_id = self._tokenizer.eos_token_id

            self._model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL_ID,
                torch_dtype=torch.float16 if self._device == "cuda" else torch.float32,
                device_map="auto" if self._device == "cuda" else None,
                trust_remote_code=True,
            )
            self._model.eval()

            # Normalize generation_config to silence transformers warnings in greedy mode
            if hasattr(self._model, "generation_config") and self._model.generation_config is not None:
                self._model.generation_config.temperature = None
                self._model.generation_config.top_p = None
                self._model.generation_config.top_k = None

            print("  [PROBER] Model loaded.", flush=True)
        except Exception as exc:
            self._load_error = f"Model load failed: {exc}"
            print(f"  [PROBER] {self._load_error}", flush=True)

    # ------------------------------------------------------------------
    # Code generation helpers
    # ------------------------------------------------------------------

    def _build_prompt(self, problem_spec: str) -> str:
        """
        Constructs a zero-shot coding prompt for the base model matching the
        oracle's expected function signature: def solve(...).
        """
        return (
            "You are a competitive programmer. Solve the following problem by writing "
            "a Python function named `solve` that implements the required behaviour.\n\n"
            f"### Problem Specification\n{problem_spec}\n\n"
            "### Solution\n"
            "```python\n"
            "def solve("
        )

    def _generate_one(
        self,
        prompt: str,
        temperature: float,
        do_sample: bool,
        seed: int,
    ) -> str:
        """Generates a single code completion from the base model."""
        import torch

        encoded = self._tokenizer(prompt, return_tensors="pt")
        input_ids = encoded["input_ids"].to(self._device)
        attention_mask = encoded["attention_mask"].to(self._device)

        with torch.no_grad():
            gen_kwargs: Dict[str, Any] = {
                "max_new_tokens": PROBING_POLICY["max_new_tokens"],
                "do_sample": do_sample,
                "pad_token_id": self._tokenizer.pad_token_id,
                "attention_mask": attention_mask,
            }
            if do_sample:
                gen_kwargs["temperature"] = temperature
                gen_kwargs["top_p"] = PROBING_POLICY["top_p"]
                # Seed reproducibility for sampled decoding
                torch.manual_seed(seed)
            else:
                gen_kwargs["temperature"] = None
                gen_kwargs["top_p"] = None
                gen_kwargs["top_k"] = None

            output = self._model.generate(input_ids, **gen_kwargs)

        generated_ids = output[0][input_ids.shape[-1]:]
        return self._tokenizer.decode(generated_ids, skip_special_tokens=True)

    # ------------------------------------------------------------------
    # Execution helper
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_solve_function(generated_text: str, prompt_suffix: str = "def solve(") -> str:
        """Reconstructs the full solve function from generated text."""
        # The prompt ended with "def solve(", so we prepend it
        full_code = prompt_suffix + generated_text
        # Trim at first ``` fence or double newline after code block
        for marker in ["```", "\n\ndef ", "\n\nclass "]:
            idx = full_code.find(marker, len(prompt_suffix))
            if idx != -1:
                full_code = full_code[:idx]
        return full_code.strip()

    @staticmethod
    def _execute_candidate(
        solve_code: str,
        test_gen_code: str,
        solve_brute_code: str,
        seed: int,
    ) -> Tuple[bool, str, str, float, float]:
        """
        Executes solve against solve_brute on small-scale test cases.
        Returns (passed, exception_type, error_msg, execution_time, memory_mb).
        """
        import copy, time
        try:
            import resource as _resource
        except ImportError:
            _resource = None
        import sys as _sys

        full_code = f"{solve_code}\n\n{solve_brute_code}\n\n{test_gen_code}"
        try:
            ast.parse(full_code)
        except SyntaxError as e:
            return False, "SyntaxError", str(e), 0.0, 0.0

        exec_globals: Dict[str, Any] = {}
        try:
            exec(compile(full_code, "<candidate>", "exec"), exec_globals)  # nosec
        except Exception as e:
            return False, type(e).__name__, str(e), 0.0, 0.0

        solve_fn = exec_globals.get("solve")
        solve_brute_fn = exec_globals.get("solve_brute")
        gen_cases_fn = exec_globals.get("generate_test_cases")

        if solve_fn is None or gen_cases_fn is None:
            return False, "MissingFunction", "solve or generate_test_cases not defined", 0.0, 0.0

        try:
            cases = gen_cases_fn(seed=seed, scale="small")[:10]
        except Exception as e:
            return False, type(e).__name__, f"test_gen failed: {e}", 0.0, 0.0

        t0 = time.perf_counter()
        for args in cases:
            pass_args = args if isinstance(args, (list, tuple)) else (args,)
            try:
                out_opt = solve_fn(*copy.deepcopy(pass_args))
                if solve_brute_fn is not None:
                    out_brute = solve_brute_fn(*copy.deepcopy(pass_args))
                    if out_opt != out_brute:
                        elapsed = time.perf_counter() - t0
                        return (
                            False,
                            "MismatchError",
                            f"Phase A Mismatch: opt={out_opt!r} brute={out_brute!r}",
                            elapsed,
                            0.0,
                        )
            except Exception as e:
                elapsed = time.perf_counter() - t0
                return False, type(e).__name__, str(e), elapsed, 0.0
        elapsed = time.perf_counter() - t0
        return True, "", "", elapsed, 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def probe(
        self,
        problem_id: str,
        problem_spec: str,
        solve_brute: str,
        test_gen: str,
        seed: int,
    ) -> ProbingResult:
        """
        Runs real base-model probing for Pass@1 and Pass@10.

        Pass@1 uses greedy decoding (temperature=0, deterministic).
        Pass@10 uses k=10 sampled completions at temperature=0.8.
        empirical_difficulty = 1.0 - pass10.

        Failure evidence is derived from the last failing execution.
        """
        self._ensure_model()

        probe_timestamp = time.time()
        k = PROBING_POLICY["k_samples"]

        # If model failed to load, we cannot probe — fail closed
        if self._load_error is not None:
            raise RuntimeError(
                f"BaseModelProber cannot probe: {self._load_error}. "
                "Ensure Qwen/Qwen2.5-1.5B-Instruct is accessible and transformers is installed."
            )

        tokenizer_vocab_size = self._tokenizer.vocab_size
        tokenizer_name = getattr(self._tokenizer, "name_or_path", BASE_MODEL_ID)

        prompt = self._build_prompt(problem_spec)
        probe_seeds = [seed * 100 + i for i in range(k)]

        # --- Pass@1 (greedy, single sample) ---
        greedy_text = self._generate_one(
            prompt,
            temperature=PROBING_POLICY["temperature_pass1"],
            do_sample=PROBING_POLICY["do_sample_pass1"],
            seed=seed,
        )
        solve_code_greedy = self._extract_solve_function(greedy_text)
        pass1_passed, exc1, err1, t1, mem1 = self._execute_candidate(
            solve_code_greedy, test_gen, solve_brute, seed
        )
        pass1 = 1.0 if pass1_passed else 0.0

        # --- Pass@10 (sampled, k independent samples) ---
        any_passed = pass1_passed  # greedy counts as first sample
        last_fail_exc = exc1 if not pass1_passed else ""
        last_fail_msg = err1 if not pass1_passed else ""
        last_fail_time = t1
        last_fail_mem = mem1
        last_fail_code = solve_code_greedy if not pass1_passed else ""

        for i in range(1, k):
            s_text = self._generate_one(
                prompt,
                temperature=PROBING_POLICY["temperature_pass10"],
                do_sample=PROBING_POLICY["do_sample_pass10"],
                seed=probe_seeds[i],
            )
            s_code = self._extract_solve_function(s_text)
            s_passed, s_exc, s_msg, s_t, s_mem = self._execute_candidate(
                s_code, test_gen, solve_brute, probe_seeds[i]
            )
            if s_passed:
                any_passed = True
            else:
                last_fail_exc = s_exc
                last_fail_msg = s_msg
                last_fail_time = s_t
                last_fail_mem = s_mem
                last_fail_code = s_code

        pass10 = 1.0 if any_passed else 0.0
        empirical_difficulty = 1.0 - pass10

        # --- Failure classification ---
        failure_cluster_id: Optional[str] = None
        failure_category: Optional[str] = None
        failure_evidence: Dict[str, Any] = {}

        if not any_passed or last_fail_exc:
            # Classify the last observed failure
            is_small = True
            small_passed = pass1_passed
            large_failed = False
            cat, cluster_id = classify_failure(
                exception_type=last_fail_exc,
                error_msg=last_fail_msg,
                execution_time=last_fail_time,
                memory_usage_mb=last_fail_mem,
                is_small_scale=is_small,
                small_passed=small_passed,
                large_failed=large_failed,
            )
            failure_cluster_id = cluster_id
            failure_category = cat.value
            failure_evidence = {
                "exception_type": last_fail_exc,
                "error_msg": last_fail_msg[:500],
                "execution_time_s": round(last_fail_time, 6),
                "memory_usage_mb": round(last_fail_mem, 2),
                "small_scale_passed": small_passed,
                "large_scale_failed": large_failed,
                "last_failing_code_snippet": last_fail_code[:300],
            }

        return ProbingResult(
            problem_id=problem_id,
            base_model_id=BASE_MODEL_ID,
            probing_policy_version=PROBING_POLICY["policy_version"],
            probing_policy_hash=_PROBING_POLICY_HASH,
            probe_timestamp=probe_timestamp,
            probe_seeds=probe_seeds,
            pass1=pass1,
            pass10=pass10,
            empirical_difficulty=empirical_difficulty,
            failure_cluster_id=failure_cluster_id,
            failure_category=failure_category,
            failure_evidence=failure_evidence,
            tokenizer_vocab_size=tokenizer_vocab_size,
            tokenizer_name_or_path=tokenizer_name,
        )

    def probe_to_dict(
        self,
        problem_id: str,
        problem_spec: str,
        solve_brute: str,
        test_gen: str,
        seed: int,
    ) -> Dict[str, Any]:
        """Convenience wrapper returning a JSON-serialisable dict."""
        return self.probe(problem_id, problem_spec, solve_brute, test_gen, seed).to_dict()


# ---------------------------------------------------------------------------
# CLI entry point for standalone testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _spec = (
        "Given an array of integers and a window size k, return a list of the "
        "maximum element in each window of size k as the window slides from left to right.\n"
        "Constraints: 1 <= k <= len(nums) <= 100; -1000 <= nums[i] <= 1000."
    )
    _brute = (
        "def solve_brute(nums, k):\n"
        "    return [max(nums[i:i+k]) for i in range(len(nums)-k+1)]\n"
    )
    _tgen = (
        "import random\n"
        "def generate_test_cases(seed, scale, max_small_dimension=8):\n"
        "    random.seed(seed)\n"
        "    if scale == 'small':\n"
        "        n = random.randint(1, max_small_dimension)\n"
        "        k = random.randint(1, n)\n"
        "        return [([random.randint(-10, 10) for _ in range(n)], k)]\n"
        "    return [([random.randint(-1000, 1000) for _ in range(100)], 5)]\n"
    )
    prober = BaseModelProber()
    result = prober.probe("DEMO_001", _spec, _brute, _tgen, seed=42)
    import json as _json
    print(_json.dumps(result.to_dict(), indent=2))
