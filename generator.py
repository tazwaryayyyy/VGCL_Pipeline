import os
import random
import time
import re
from typing import Any, Dict
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    timeout=30.0,
)

_MODEL_PRIORITY = [
    "gemini-3.6-flash",
    "gemini-3.1-flash-lite",
    "gemini-3.5-flash-lite",
    "gemini-flash-latest",
]


def parse_delimiters(raw_text: str) -> Dict[str, Any]:
    """Parses strict top-level delimiter sections from LLM generation."""
    headers = [
        "### PROBLEM_SPEC",
        "### SOLVE_OPTIMIZED",
        "### SOLVE_BRUTE",
        "### TEST_GENERATOR",
    ]

    pos = []
    for h in headers:
        idx = raw_text.find(h)
        if idx == -1:
            return {"valid": False, "error": f"Missing required block header: {h}"}
        pos.append((idx, h))

    pos.sort()

    extracted = {}
    for i in range(len(pos)):
        start_idx = pos[i][0] + len(pos[i][1])
        end_idx = pos[i + 1][0] if i + 1 < len(pos) else len(raw_text)
        content = raw_text[start_idx:end_idx].strip()

        if content.startswith("```python"):
            content = content[9:]
        elif content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]

        header_key = pos[i][1].replace("### ", "").lower()
        if header_key == "problem_spec":
            extracted["spec"] = content.strip()
        elif header_key == "solve_optimized":
            extracted["solve_opt"] = content.strip()
        elif header_key == "solve_brute":
            extracted["solve_brute"] = content.strip()
        elif header_key == "test_generator":
            extracted["test_gen"] = content.strip()

    return {
        "valid": True,
        "spec": extracted.get("spec", ""),
        "solve_opt": extracted.get("solve_opt", ""),
        "solve_brute": extracted.get("solve_brute", ""),
        "test_gen": extracted.get("test_gen", ""),
    }


def _call_llm_with_backoff(system_prompt: str, user_prompt: str) -> str:
    """Calls Gemini API with failover across priority models for quota and model errors."""
    last_exception = ""
    for model in _MODEL_PRIORITY:
        max_attempts = 3    # 3 tries per model before failing over.
        base_delay = 5.0    # 5s base; absorbs typical quota-reset windows.

        for attempt in range(max_attempts):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.4,
                )
                return response.choices[0].message.content or ""

            except Exception as e:
                last_exception = f"[{model}] {e}"
                err_str = str(e).lower()
                is_quota = any(kw in err_str for kw in ["429", "resource_exhausted", "quota", "rate limit"])

                if is_quota and attempt < max_attempts - 1:
                    sleep_s = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    print(
                        f"  [RATE-LIMIT] {model} quota hit (attempt {attempt + 1}/{max_attempts}). "
                        f"Backing off {sleep_s:.1f}s ...",
                        flush=True,
                    )
                    time.sleep(sleep_s)
                    continue
                # Non-quota error or exhausted retries: fail over to next model.
                break

    return f"LLM Generation Exception: All priority models failed. Last error: {last_exception}"


def generate_candidate_raw(
    paradigm: str,
    seed: int,
    attempt: int = 1,
    last_error: str = "",
    last_code: str = "",
    failing_input: str = "",
) -> str:
    """Generates candidate algorithmic code packages using enforced system contracts."""

    system_prompt = (
        "You are a principal competitive programming problem architect. Your task is to design "
        "a self-contained, mathematically sound algorithmic problem package.\n\n"
        "### CRITICAL SYSTEM CONTRACTS (VIOLATION CAUSES HARD SYSTEM REJECTION)\n\n"
        "1. FUNCTION SIGNATURE CONTRACT:\n"
        "   - 'solve' and 'solve_brute' MUST accept the EXACT same positional arguments.\n"
        "   - Example: If 'solve(grid, k)' takes 2 arguments, 'solve_brute(grid, k)' MUST take 2 arguments.\n"
        "   - 'generate_test_cases(seed: int, scale: str)' MUST return a list of tuples, where EVERY tuple "
        "matches the positional arguments of 'solve' 1-to-1.\n"
        "   - NEVER return placeholder singletons like [(seed,)] or [seed]. Return real, fully formed input tuples.\n\n"
        "2. TEST GENERATOR SCALE CONTRACT:\n"
        "   - Line 1 of 'generate_test_cases' MUST be: random.seed(seed)\n"
        "   - You MUST explicitly implement scale branching inside the code:\n"
        "       if scale == 'small': generate cases where N is in [1, 15]\n"
        "       elif scale == 'large': generate cases where N is in [10^4, 10^5]\n"
        "   - Generate at least 5 distinct test tuples for scale='small' and scale='large'.\n\n"
        "3. SOLVE VS BRUTE DIFFERENTIATION CONTRACT:\n"
        "   - 'solve' MUST implement the optimal algorithm for the paradigm.\n"
        "   - 'solve_brute' MUST implement a naive/brute-force approach (e.g., exponential search, "
        "nested loops, or unoptimized recursion).\n"
        "   - 'solve_brute' MUST NOT be a copy-paste or simple wrapper of 'solve'.\n\n"
        "4. OUTPUT FORMAT CONTRACT:\n"
        "   - Do NOT wrap response blocks in markdown triple backticks (```python ... ```).\n"
        "   - Output EXACTLY four blocks using these exact section headers:\n\n"
        "### PROBLEM_SPEC\n"
        "<Problem description, input/output formats, constraints>\n\n"
        "### SOLVE_OPTIMIZED\n"
        "def solve(...):\n"
        "    ...\n\n"
        "### SOLVE_BRUTE\n"
        "def solve_brute(...):\n"
        "    ...\n\n"
        "### TEST_GENERATOR\n"
        "import random\n"
        "def generate_test_cases(seed: int, scale: str) -> list:\n"
        "    ...\n"
    )

    if attempt == 1:
        user_prompt = (
            f"Target Paradigm: {paradigm}\n"
            f"Seed: {seed}\n\n"
            f"Generate a new problem package for '{paradigm}' matching all system contracts."
        )
    else:
        # Truncate error payload if excessively long to protect context window
        clean_error = last_error[-1500:] if len(last_error) > 1500 else last_error
        
        user_prompt = (
            f"RETRY ATTEMPT {attempt}/3 FOR PARADIGM '{paradigm}' (Seed {seed}).\n\n"
            f"YOUR PREVIOUS CODE FAILED THE AUDIT WITH THE FOLLOWING ERROR:\n"
            f"--------------------------------------------------\n"
            f"{clean_error}\n"
            f"--------------------------------------------------\n\n"
        )
        if failing_input:
            user_prompt += f"Failing Test Case Input Payload:\n{failing_input}\n\n"

        user_prompt += (
            f"Previous Defective Code Attempt:\n"
            f"```\n{last_code}\n```\n\n"
            f"Fix the bug causing the failure above. Ensure 'generate_test_cases' returns valid tuples "
            f"matching 'solve' arguments and branches properly on scale='small' vs scale='large'."
        )

    return _call_llm_with_backoff(system_prompt, user_prompt)


generate_candidate = generate_candidate_raw