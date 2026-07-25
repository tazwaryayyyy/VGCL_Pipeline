import os
import re
import time
import random
from typing import Any, Dict
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    timeout=60.0,
)

# Model priority list: confirmed working free-tier models, most-capable first
_MODEL_PRIORITY = [
    "gemini-3.6-flash",
    "gemini-3.1-flash-lite",
    "gemini-3.5-flash-lite",
    "gemini-3-flash-preview",
    "gemini-3.1-flash-lite-preview",
    "gemini-flash-latest",
    "gemini-flash-lite-latest",
]


def parse_delimiters(raw_text: str) -> Dict[str, Any]:
    """Parses strict top-level delimiter sections from LLM generation without lookahead regex bugs."""
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

        # Strip markdown code fences if present inside section
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
    """
    Calls the Gemini OpenAI-compat endpoint with exponential backoff on 429/quota errors.
    Falls back through _MODEL_PRIORITY if a model stays exhausted after max_attempts.
    Returns raw content string, or an error sentinel starting with 'LLM Generation Exception:'.
    """
    for model in _MODEL_PRIORITY:
        max_attempts = 3
        base_delay = 15.0  # seconds

        for attempt in range(max_attempts):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.3,
                )
                return response.choices[0].message.content or ""

            except Exception as e:
                err_str = str(e).lower()
                is_quota = any(kw in err_str for kw in [
                    "429", "resource_exhausted", "quota", "rate limit", "ratelimit",
                ])

                if is_quota and attempt < max_attempts - 1:
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 5)
                    print(
                        f"  [!] {model} quota hit (attempt {attempt + 1}/{max_attempts}). "
                        f"Backing off {delay:.1f}s..."
                    )
                    time.sleep(delay)
                    continue

                if is_quota:
                    # Exhausted retries on this model — cascade to next
                    print(
                        f"  [!] {model} quota exhausted after {max_attempts} attempts. "
                        f"Cascading to next model..."
                    )
                    break  # break inner loop → continue outer model loop

                # Non-quota error: fail immediately, don't try other models
                return f"LLM Generation Exception: [{model}] {e}"

    return "LLM Generation Exception: All models in priority list are quota-exhausted."


def generate_candidate_raw(
    paradigm: str,
    seed: int,
    attempt: int = 1,
    last_error: str = "",
    last_code: str = "",
    failing_input: str = "",
) -> str:
    """Generates candidate raw text using strict context resets and LLM completion."""
    system_prompt = (
        "You are an expert algorithmic problem designer and competitive programmer. "
        "Your task is to generate a fully self-contained algorithmic problem, including an optimal solution, "
        "a canonical brute-force solution, and a deterministic test case generator.\n\n"
        "You MUST follow this EXACT delimiter format without exception. "
        "Do not wrap blocks in backticks or markdown fences outside the section delimiters.\n\n"
        "### PROBLEM_SPEC\n"
        "Target Paradigm: {paradigm}\n"
        "Problem Name: {problem_name}\n\n"
        "Description:\n"
        "<Detailed problem statement. Clear specification of inputs, constraints, and expected output format.>\n\n"
        "Input Constraints:\n"
        "- Small Scale: N in [1, 15]\n"
        "- Large Scale: N in [10^4, 10^5]\n\n"
        "### SOLVE_OPTIMIZED\n"
        "import math\n"
        "import collections\n"
        "import heapq\n"
        "import bisect\n"
        "import itertools\n\n"
        "def solve(*args):\n"
        "    \"\"\"\n"
        "    Optimal production code.\n"
        "    Target Complexity: O(N) or O(N log N). Must pass within 2.0s for N = 10^5.\n"
        "    Must accept positional arguments exactly as returned by generate_test_cases().\n"
        "    \"\"\"\n"
        "    pass\n\n"
        "### SOLVE_BRUTE\n"
        "def solve_brute(*args):\n"
        "    \"\"\"\n"
        "    Canonical brute-force. Must NOT call solve(). Must return same type as solve().\n"
        "    \"\"\"\n"
        "    pass\n\n"
        "### TEST_GENERATOR\n"
        "import random\n\n"
        "def generate_test_cases(seed: int, scale: str) -> list:\n"
        "    \"\"\"\n"
        "    Returns a list of argument tuples. scale='small' -> N in [1,15], scale='large' -> N in [10^4,10^5].\n"
        "    Must be deterministic given the same seed.\n"
        "    \"\"\"\n"
        "    pass\n\n"
        "RULES:\n"
        "1. Do NOT import os, sys, subprocess, threading, or multiprocessing.\n"
        "2. Do NOT mutate global state across calls.\n"
        "3. Ensure solve() and solve_brute() return identical data structures for all inputs.\n"
        "4. Critical: solve_brute() MUST NOT call solve().\n"
        "5. generate_test_cases() must return a list of tuples (or single values) as positional args to solve()."
    )

    if attempt == 1:
        user_prompt = f"Generate an algorithmic problem for paradigm '{paradigm}' using seed {seed}."
    else:
        user_prompt = (
            f"RETRY ATTEMPT {attempt}/3 FOR PARADIGM '{paradigm}'.\n"
            f"Your previous attempt failed with the following error:\n{last_error}\n"
        )
        if failing_input:
            user_prompt += f"Failing Input Payload:\n{failing_input[:500]}\n\n"

        user_prompt += (
            f"Previous Failed Code snippet:\n{last_code[:400]}\n\n"
            f"Fix the bug and emit ALL required blocks again using the EXACT delimiter format."
        )

    return _call_llm_with_backoff(system_prompt, user_prompt)


# Alias so any legacy import of generate_candidate still works
generate_candidate = generate_candidate_raw
