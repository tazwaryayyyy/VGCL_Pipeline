import re
import traceback
from typing import Any, Dict, Optional, Tuple

from generator import generate_candidate_raw, parse_delimiters
from verifier import run_full_verification


def sanitize_traceback(tb_str: str) -> str:
    """Strips absolute file paths and framework stack frames."""
    lines = tb_str.strip().split("\n")
    sanitized = []
    for line in lines:
        clean_line = re.sub(r'File ["\'].*?[\\/]([^\\/]+)["\']', r'File "\1"', line)
        if "site-packages" in line or "importlib" in line:
            continue
        sanitized.append(clean_line)

    return "\n".join(sanitized[-6:])


def execute_retry_loop(
    paradigm: str, seed: int, max_retries: int = 3
) -> Tuple[bool, Optional[Dict[str, Any]], str]:
    """Runs generation retry loop with untruncated code context pass-through."""
    last_error = ""
    latest_code_attempt = ""
    last_failing_input = ""

    for attempt in range(1, max_retries + 1):
        try:
            raw_response = generate_candidate_raw(
                paradigm=paradigm,
                seed=seed,
                attempt=attempt,
                last_error=last_error,
                last_code=latest_code_attempt,
                failing_input=last_failing_input,
            )

            parsed = parse_delimiters(raw_response)
            if not parsed["valid"]:
                last_error = f"ParseError: {parsed['error']}"
                latest_code_attempt = raw_response
                last_failing_input = ""
                continue

            full_code = (
                f"{parsed['solve_opt']}\n\n"
                f"{parsed['solve_brute']}\n\n"
                f"{parsed['test_gen']}"
            )
            latest_code_attempt = full_code

            verified, ver_msg, failing_input = run_full_verification(full_code, seed=seed)

            if verified:
                candidate_data = {
                    "paradigm": paradigm,
                    "seed": seed,
                    "spec": parsed["spec"],
                    "code": full_code,
                    "solve_opt": parsed["solve_opt"],
                    "solve_brute": parsed["solve_brute"],
                    "test_gen": parsed["test_gen"],
                    "attempts_used": attempt,
                }
                return True, candidate_data, "SUCCESS"

            last_error = sanitize_traceback(ver_msg)
            last_failing_input = failing_input if failing_input else ""

        except Exception as e:
            tb = traceback.format_exc()
            last_error = sanitize_traceback(f"{type(e).__name__}: {e}\n{tb}")
            last_failing_input = ""

    return False, None, f"PERMANENT_REJECT: Exceeded {max_retries} retries. Last error: {last_error}"
