import os
import random
import time
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
        max_attempts = 2
        base_delay = 2.0

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
                    time.sleep(base_delay + random.uniform(0, 1))
                    continue
                # Fail over to next model priority on persistent errors or non-429 exceptions
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
    """Generates candidate code using full code context pass-through on retries."""
    system_prompt_template = (
        "You are an expert algorithmic problem designer. Generate a self-contained algorithmic problem, "
        "including an optimal solution, canonical brute-force, and deterministic test generator.\n\n"
        "### PROBLEM_SPEC\n"
        "Target Paradigm: {paradigm}\n"
        "Seed: {seed}\n\n"
        "Description:\n<Problem description>\n\n"
        "Input Constraints:\n- Small Scale: N in [1, 15]\n- Large Scale: N in [10^4, 10^5]\n\n"
        "### SOLVE_OPTIMIZED\n"
        "def solve(*args):\n    pass\n\n"
        "### SOLVE_BRUTE\n"
        "def solve_brute(*args):\n    pass\n\n"
        "### TEST_GENERATOR\n"
        "import random\n"
        "def generate_test_cases(seed: int, scale: str) -> list:\n    pass\n"
    )

    system_prompt = system_prompt_template.format(paradigm=paradigm, seed=seed)

    if attempt == 1:
        user_prompt = f"Generate an algorithmic problem for paradigm '{paradigm}' using seed {seed}."
    else:
        user_prompt = (
            f"RETRY ATTEMPT {attempt}/3 FOR PARADIGM '{paradigm}'.\n"
            f"Previous Error:\n{last_error}\n\n"
        )
        if failing_input:
            user_prompt += f"Failing Input Payload:\n{failing_input}\n\n"

        user_prompt += (
            f"Previous Complete Code Attempt:\n{last_code}\n\n"
            f"Fix the defect and output ALL required blocks using the exact delimiter format."
        )

    return _call_llm_with_backoff(system_prompt, user_prompt)


generate_candidate = generate_candidate_raw
