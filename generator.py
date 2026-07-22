import os
import re
from typing import Any, Dict
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    timeout=60.0,
)


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

        # Clean markdown code fences if present inside section
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
        "You MUST follow this EXACT delimiter format without exception. Do not wrap blocks in backticks or markdown fences outside the section delimiters.\n\n"
        "### PROBLEM_SPEC\n"
        "Target Paradigm: {paradigm}\n"
        "Problem Name: {problem_name}\n\n"
        "Description:\n"
        "<Detailed problem statement>\n\n"
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
        "    pass\n\n"
        "### SOLVE_BRUTE\n"
        "def solve_brute(*args):\n"
        "    pass\n\n"
        "### TEST_GENERATOR\n"
        "import random\n\n"
        "def generate_test_cases(seed: int, scale: str) -> list:\n"
        "    pass\n\n"
        "RULES:\n"
        "1. Do NOT import os, sys, subprocess, threading, or multiprocessing.\n"
        "2. Do NOT mutate global state across calls.\n"
        "3. Ensure solve() and solve_brute() return identical data structures.\n"
        "4. Critical: solve_brute() MUST NOT call solve()."
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

    try:
        response = client.chat.completions.create(
            model="gemini-2.0-flash",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
        )
        return response.choices[0].message.content or ""
    except Exception as e:
        return f"LLM Generation Exception: {e}"
