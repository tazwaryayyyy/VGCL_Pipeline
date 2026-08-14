"""syntax_repair.py

Lightweight pre-verification syntax validation and single-attempt repair layer
for VGCL Phase 0.

Pipeline position:
    generated code
        ↓
    ast.parse()          ← fast, in-process, no subprocess overhead
        ↓ (SyntaxError)
    repair_attempt()     ← one attempt: strip markdown fences, normalise
                           trailing whitespace; never alters algorithmic logic
        ↓
    ast.parse()          ← second validation
        ↓ (SyntaxError)
    SyntaxDiagnostic(passed=False, ...)
        ↓ (success, either pass)
    SyntaxDiagnostic(passed=True, ...)  →  caller proceeds to oracle

Design invariants:
- Repair is syntax-only.  It never removes statements, changes variable names,
  modifies control flow, or simplifies algorithms.
- Every repair action is described in ``repair_applied`` for auditing.
- The oracle remains the definitive correctness gate; this layer only avoids
  spending a 4–12 s subprocess budget on code that cannot even be parsed.
- All classification categories are fixed string literals so downstream
  machine-readable consumers are stable.
"""

import ast
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

# Canonical error category strings (machine-readable, stable identifiers)
_CAT_NONE = "none"
_CAT_INDENTATION = "indentation"
_CAT_UNEXPECTED_EOF = "unexpected_eof"
_CAT_MISSING_BLOCK = "missing_block"
_CAT_OTHER = "other"


@dataclass
class SyntaxDiagnostic:
    """
    Structured result of one syntax validation (and optional repair) attempt.

    Attributes:
        passed:           True if the code parses successfully as valid Python.
        repaired:         True if at least one repair transformation was applied
                          before the second ``ast.parse`` succeeded.
        repair_applied:   Human-readable description of the transformation(s)
                          applied, or ``"none"`` if no repair was attempted or
                          no transformation was necessary.
        error_category:   One of ``"none"`` | ``"indentation"`` |
                          ``"unexpected_eof"`` | ``"missing_block"`` |
                          ``"other"``.  ``"none"`` when ``passed`` is True.
        error_line:       Source line number of the syntax error reported by
                          the parser, or ``None`` if unavailable.
        error_msg:        Raw error message string from the parser, or
                          ``""`` when ``passed`` is True.
    """

    passed: bool
    repaired: bool
    repair_applied: str
    error_category: str
    error_line: Optional[int]
    error_msg: str


def _classify_syntax_error(exc: SyntaxError) -> str:
    """Classifies a SyntaxError into a stable category string."""
    msg = str(exc).lower()
    if "indent" in msg or "unindent" in msg or "unexpected indent" in msg:
        return _CAT_INDENTATION
    if "unexpected eof" in msg or "eof while" in msg or "was never closed" in msg:
        return _CAT_UNEXPECTED_EOF
    if "expected an indented block" in msg or "expected ':'" in msg:
        return _CAT_MISSING_BLOCK
    return _CAT_OTHER


def _strip_markdown_fences(code: str) -> tuple[str, bool]:
    """
    Removes residual markdown code fences from LLM output.

    The generation prompt explicitly forbids triple-backtick wrapping, but
    models occasionally emit them.  This strips:
      - Leading `` ```python `` or `` ``` `` lines (with optional trailing space)
      - Trailing `` ``` `` lines

    Returns:
        (stripped_code, changed): the cleaned string and whether anything changed.
    """
    lines = code.split("\n")
    changed = False

    # Strip leading fence line
    if lines and re.match(r"^```(python)?\s*$", lines[0].strip()):
        lines = lines[1:]
        changed = True

    # Strip trailing fence line
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
        changed = True

    return "\n".join(lines), changed


def _strip_trailing_whitespace(code: str) -> tuple[str, bool]:
    """
    Removes trailing whitespace from each line and trailing blank lines.

    This is a safe, semantics-preserving normalisation.

    Returns:
        (normalised_code, changed)
    """
    lines = code.split("\n")
    normalised = [ln.rstrip() for ln in lines]
    # Remove trailing blank lines
    while normalised and normalised[-1] == "":
        normalised.pop()
    result = "\n".join(normalised)
    changed = result != code
    return result, changed


def _attempt_repair(code: str) -> tuple[str, str]:
    """
    Applies all safe, syntax-only repair transformations in sequence.

    Each transformation is independent; all are applied in one pass.
    Returns:
        (repaired_code, description): repaired code and a loggable description
        of what changed, or ``"none"`` if nothing changed.
    """
    applied: list[str] = []

    code, fence_changed = _strip_markdown_fences(code)
    if fence_changed:
        applied.append("stripped_markdown_fences")

    code, ws_changed = _strip_trailing_whitespace(code)
    if ws_changed:
        applied.append("stripped_trailing_whitespace")

    description = ", ".join(applied) if applied else "none"
    return code, description


def check_and_repair(code: str) -> tuple[SyntaxDiagnostic, str]:
    """
    Validates ``code`` as Python, applies at most one repair attempt on failure,
    and returns a structured diagnostic together with the (possibly repaired) code.

    This function is the sole public API of this module.  It is designed to be
    called on the concatenated ``full_code`` string immediately before submitting
    to the verification oracle.

    Pipeline:
        1. ``ast.parse(code)``  →  if OK: return ``passed=True, repaired=False``
        2. ``_attempt_repair(code)``  →  produces candidate repaired code
        3. ``ast.parse(repaired)``  →  if OK: return ``passed=True, repaired=True``
        4. Classify the remaining error  →  return ``passed=False``

    Args:
        code: The full concatenated Python source string to validate.

    Returns:
        A tuple of:
        - ``SyntaxDiagnostic``: structured result (see class docstring).
        - ``str``: the code string to use downstream.  When ``passed=True``,
          this is either the original code (if no repair was needed) or the
          repaired version.  When ``passed=False``, this is the original
          unmodified code so the oracle / quarantine record the original LLM
          output faithfully.
    """
    # Pass 1: try the code as-is
    try:
        ast.parse(code)
        return (
            SyntaxDiagnostic(
                passed=True,
                repaired=False,
                repair_applied="none",
                error_category=_CAT_NONE,
                error_line=None,
                error_msg="",
            ),
            code,
        )
    except SyntaxError as first_exc:
        first_category = _classify_syntax_error(first_exc)
        first_line: Optional[int] = first_exc.lineno
        first_msg = str(first_exc)

    # Pass 2: attempt repair, then re-validate
    repaired_code, repair_description = _attempt_repair(code)

    if repaired_code == code:
        # Repair made no changes; skip second parse — result cannot differ.
        return (
            SyntaxDiagnostic(
                passed=False,
                repaired=False,
                repair_applied="none",
                error_category=first_category,
                error_line=first_line,
                error_msg=first_msg,
            ),
            code,
        )

    try:
        ast.parse(repaired_code)
        return (
            SyntaxDiagnostic(
                passed=True,
                repaired=True,
                repair_applied=repair_description,
                error_category=_CAT_NONE,
                error_line=None,
                error_msg="",
            ),
            repaired_code,
        )
    except SyntaxError as second_exc:
        second_category = _classify_syntax_error(second_exc)
        second_line: Optional[int] = second_exc.lineno
        second_msg = str(second_exc)
        return (
            SyntaxDiagnostic(
                passed=False,
                repaired=False,
                repair_applied=repair_description,  # log what was tried
                error_category=second_category,
                error_line=second_line,
                error_msg=second_msg,
            ),
            code,  # return original so quarantine record is faithful
        )


def check_component_syntax(code: str) -> tuple[SyntaxDiagnostic, str]:
    """Validates and applies whitespace/fence normalization to a single component."""
    return check_and_repair(code)


def identify_malformed_component(
    blocks: dict[str, str],
) -> tuple[Optional[str], Optional[SyntaxDiagnostic]]:
    """Identifies which code component fails Python syntax parsing.

    Checks components in deterministic order: solve_opt, solve_brute, test_gen.
    Returns (component_name, diagnostic) if a component fails, or (None, None) if all pass.
    """
    for comp in ("solve_opt", "solve_brute", "test_gen"):
        code = blocks.get(comp, "")
        diag, _ = check_and_repair(code)
        if not diag.passed:
            return comp, diag
    return None, None


def parse_single_component_content(raw_text: str, header: str) -> str:
    """Extracts code block from a single-component regeneration response."""
    idx = raw_text.find(header)
    if idx != -1:
        start = idx + len(header)
        # Find next header or end
        headers = ["### PROBLEM_SPEC", "### SOLVE_OPTIMIZED", "### SOLVE_BRUTE", "### TEST_GENERATOR"]
        end_indices = [raw_text.find(h, start) for h in headers if raw_text.find(h, start) != -1]
        end = min(end_indices) if end_indices else len(raw_text)
        content = raw_text[start:end].strip()
    else:
        content = raw_text.strip()

    if content.startswith("```python"):
        content = content[9:]
    elif content.startswith("```"):
        content = content[3:]
    if content.endswith("```"):
        content = content[:-3]
    return content.strip()


def regenerate_syntax_component(
    component_name: str,
    blocks: dict[str, str],
    cluster_id: str,
    subtype: str,
    seed: int,
    llm_caller,
    original_error: str = "",
) -> tuple[bool, dict[str, str], dict[str, Any], int]:
    """Performs exactly ONE targeted component-level LLM regeneration.

    Never modifies problem spec, cluster, subtype, seed, or valid components.

    Returns:
        (success, updated_blocks, retry_provenance, api_calls_made)
    """
    spec = blocks.get("spec", "")
    header = ""

    if component_name == "solve_opt":
        header = "### SOLVE_OPTIMIZED"
        system_prompt = (
            "You are a principal competitive programming algorithm architect.\n"
            "The previous optimal solution for this problem had a syntax error and failed parsing.\n"
            "Write ONLY a syntactically valid Python optimal solution for the problem specification below.\n"
            "Ensure valid Python syntax, correct indentation, and complete function definitions.\n\n"
            f"### CLUSTER TARGET: {cluster_id}\n"
            f"Subtype Stratification: {subtype}\n\n"
            "### OUTPUT FORMAT:\n"
            "### SOLVE_OPTIMIZED\n"
            "def solve(...):\n"
            "    ...\n"
        )
        user_prompt = (
            f"Here is the problem specification:\n\n{spec}\n\n"
            f"Previous parse error: {original_error}\n"
            f"Regenerate 'solve' in valid Python for Seed {seed}."
        )
    elif component_name == "solve_brute":
        header = "### SOLVE_BRUTE"
        system_prompt = (
            "You are an independent algorithmic verification reference generator.\n"
            "The previous brute-force reference implementation had a syntax error and failed parsing.\n"
            "Write ONLY a simple, naive, brute-force solution (solve_brute) for the provided problem spec.\n\n"
            "### STRICT BRUTE-FORCE CONSTRAINTS:\n"
            "1. MUST use exact same positional function signature and return type as described in the spec.\n"
            "2. MUST use exhaustive search, naive loops, or unoptimized recursion.\n"
            "3. MUST NOT use memoization, dynamic programming, or advanced data structures.\n"
            "4. DO NOT attempt to write optimal code.\n\n"
            "### OUTPUT FORMAT:\n"
            "### SOLVE_BRUTE\n"
            "def solve_brute(...):\n"
            "    ...\n"
        )
        user_prompt = (
            f"Here is the problem specification:\n\n{spec}\n\n"
            f"Previous parse error: {original_error}\n"
            f"Regenerate 'solve_brute' in valid Python for Seed {seed}."
        )
    elif component_name == "test_gen":
        header = "### TEST_GENERATOR"
        system_prompt = (
            "You are an algorithmic test generator engineer.\n"
            "The previous test generator had a syntax error and failed parsing.\n"
            "Write ONLY a valid Python test case generator for the provided problem spec.\n\n"
            "### OUTPUT FORMAT:\n"
            "### TEST_GENERATOR\n"
            "import random\n"
            "def generate_test_cases(seed: int, scale: str) -> list:\n"
            "    # Line 1 MUST be: random.seed(seed)\n"
            "    # Return list of argument tuples matching 'solve' positional arguments\n"
            "    # scale=='small': N <= 8; scale=='large': N >= 100\n"
        )
        user_prompt = (
            f"Here is the problem specification:\n\n{spec}\n\n"
            f"Previous parse error: {original_error}\n"
            f"Regenerate 'generate_test_cases' in valid Python for Seed {seed}."
        )
    else:
        provenance = {
            "attempted": False,
            "component": component_name,
            "original_parse_error": original_error,
            "retry_succeeded": False,
        }
        return False, blocks, provenance, 0

    raw_text, model_name, latency, api_calls = llm_caller(system_prompt, user_prompt)
    extracted = parse_single_component_content(raw_text, header)

    diag, repaired = check_and_repair(extracted)
    if diag.passed and repaired.strip():
        new_blocks = dict(blocks)
        new_blocks[component_name] = repaired.strip()
        provenance = {
            "attempted": True,
            "component": component_name,
            "original_parse_error": original_error,
            "retry_succeeded": True,
        }
        return True, new_blocks, provenance, api_calls

    provenance = {
        "attempted": True,
        "component": component_name,
        "original_parse_error": original_error,
        "retry_succeeded": False,
    }
    return False, blocks, provenance, api_calls
