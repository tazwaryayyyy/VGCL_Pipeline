"""
audit_curriculum.py — Deep static + dynamic audit of approved_curriculum.jsonl

Checks per entry:
  Static:
    1.  BROKEN_TEST_GEN     — test_gen returns placeholder like [(seed,)]
    2.  IDENTICAL_OPT_BRUTE — solve and solve_brute are byte-for-byte the same logic
    3.  EMPTY_SPEC          — problem spec is blank
    4.  NO_SCALE_BRANCH     — test_gen does not branch on scale='small'/'large'
    5.  STUB_SOLVE          — solve body is trivially short (<3 meaningful lines)
    6.  STUB_BRUTE          — same for solve_brute
    7.  HARDCODED_RETURN    — solve always returns a constant (0, [], None, '')
    8.  MISSING_FIELDS      — any of solve_opt/solve_brute/test_gen/spec missing
    9.  SEED_UNUSED         — test_gen ignores the seed parameter (no random.seed)
    10. DUPLICATE_SEED      — same seed appears more than once in the file

  Dynamic (execute in-process, isolated):
    11. TESTGEN_EXEC_ERROR  — test_gen crashes at runtime
    12. TESTGEN_EMPTY       — test_gen returns no cases for scale='small'
    13. TESTGEN_BAD_TYPE    — test_gen returns non-list or items are not tuple/list
    14. TESTGEN_INT_ARGS    — every arg in every test case is a bare integer (no real inputs)
    15. SOLVE_EXEC_ERROR    — solve crashes on the generated test cases
    16. BRUTE_EXEC_ERROR    — solve_brute crashes on the generated test cases
    17. DIFFERENTIAL_FAIL   — solve and solve_brute return different results
    18. CONSTANT_OUTPUT     — solve returns the exact same value on all test cases (oracle blind)
    19. LARGE_MISSING       — test_gen returns same/empty cases for scale='large'
    20. LARGE_NOT_BIGGER    — large-scale inputs are not meaningfully bigger than small-scale
"""

import ast
import copy
import json
import math
import os
import sys
import traceback
from typing import Any, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from isolated_executor import SandboxInfrastructureError, audit_entry_isolated
except ImportError:
    from ..isolated_executor import SandboxInfrastructureError, audit_entry_isolated

# ── helpers ──────────────────────────────────────────────────────────────────

def _are_equal(a: Any, b: Any, tol: float = 1e-6) -> bool:
    """Recursive loose equality with float/NaN tolerance."""
    if hasattr(a, 'item'):
        a = a.item()
    if hasattr(b, 'item'):
        b = b.item()
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_are_equal(x, y, tol) for x, y in zip(a, b))
    if isinstance(a, float) and isinstance(b, float):
        if math.isnan(a) and math.isnan(b):
            return True
        return math.isclose(a, b, rel_tol=tol, abs_tol=tol)
    return a == b


def _strip_def_line(code: str, fname: str) -> str:
    """Remove 'def fname(...)' lines for body comparison."""
    return '\n'.join(
        l for l in code.strip().splitlines()
        if not l.strip().startswith(f'def {fname}')
    ).strip()


def _meaningful_lines(code: str) -> int:
    """Count non-blank, non-comment lines."""
    return sum(
        1 for l in code.splitlines()
        if l.strip() and not l.strip().startswith('#')
    )


def _has_hardcoded_return(code: str) -> bool:
    """Detect trivially constant return values via AST."""
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == 'solve':
                returns = [n for n in ast.walk(node) if isinstance(n, ast.Return)]
                if len(returns) == 1:
                    val = returns[0].value
                    if isinstance(val, (ast.Constant, ast.List, ast.Tuple)) and \
                            (not isinstance(val, (ast.List, ast.Tuple)) or not val.elts):
                        return True
    except Exception:
        pass
    return False


def _input_size(args: tuple) -> int:
    """Proxy for input size: largest collection length or integer value."""
    size = 0
    for a in args:
        if isinstance(a, (list, tuple, str)):
            size = max(size, len(a))
        elif isinstance(a, int) and not isinstance(a, bool):
            size = max(size, a)
    return size


# ── main audit ───────────────────────────────────────────────────────────────

def audit(path: str = 'data/approved_curriculum.jsonl') -> None:
    seen_seeds: dict = {}
    all_issues: list = []
    total = 0

    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for i, raw in enumerate(lines, 1):
        raw = raw.strip()
        if not raw:
            continue
        total += 1

        # ── parse ──
        try:
            e = json.loads(raw)
        except Exception as ex:
            all_issues.append((i, '?', '?', [f'JSON_PARSE_ERROR: {ex}']))
            continue

        seed  = e.get('seed')
        par   = e.get('paradigm', '?')
        spec  = e.get('spec', '')
        tgen  = e.get('test_gen', '')
        opt   = e.get('solve_opt', '')
        brute = e.get('solve_brute', '')
        code  = e.get('code', f'{opt}\n\n{brute}\n\n{tgen}')

        flags: List[str] = []

        # ── STATIC CHECKS ──────────────────────────────────────────────────

        # 1. Missing fields
        for field in ('solve_opt', 'solve_brute', 'test_gen', 'spec'):
            if not e.get(field, '').strip():
                flags.append(f'MISSING_FIELD: {field} is empty')

        # 2. Broken test generator (placeholder pattern)
        for pattern in ('return [(seed,)]', 'return [(seed)]', 'return [seed]',
                        'return []', 'pass\n', 'return None'):
            if pattern in tgen:
                flags.append(f'BROKEN_TEST_GEN: contains placeholder pattern: {pattern!r}')
                break

        # 3. No scale branching
        if tgen and ('small' not in tgen or 'large' not in tgen):
            flags.append('NO_SCALE_BRANCH: test_gen does not branch on small/large scale')

        # 4. Seed unused in test_gen
        if tgen and 'random.seed' not in tgen and 'seed' not in tgen:
            flags.append('SEED_UNUSED: test_gen never uses the seed parameter')

        # 5. Stub solve / brute
        if opt and _meaningful_lines(opt) < 4:
            flags.append(f'STUB_SOLVE: solve body has only {_meaningful_lines(opt)} meaningful lines')
        if brute and _meaningful_lines(brute) < 4:
            flags.append(f'STUB_BRUTE: solve_brute body has only {_meaningful_lines(brute)} meaningful lines')

        # 6. Hardcoded return in solve
        if opt and _has_hardcoded_return(opt):
            flags.append('HARDCODED_RETURN: solve has a single constant return value')

        # 7. Identical opt vs brute
        opt_body   = _strip_def_line(opt,   'solve')
        brute_body = _strip_def_line(brute, 'solve_brute')
        if opt_body and brute_body and opt_body == brute_body:
            flags.append('IDENTICAL_OPT_BRUTE: solve and solve_brute are copy-pastes')

        # 8. Empty spec
        if not spec.strip():
            flags.append('EMPTY_SPEC')

        # 9. Duplicate seed
        if seed is not None:
            if seed in seen_seeds:
                flags.append(f'DUPLICATE_SEED: seed {seed} already appeared at entry {seen_seeds[seed]}')
            else:
                seen_seeds[seed] = i

        # ── DYNAMIC CHECKS (Isolated Sandbox Execution) ────────────────────
        if not code.strip():
            all_issues.append((i, seed, par, flags))
            continue

        try:
            dynamic_flags = audit_entry_isolated(code=code, seed=seed or 42)
            flags.extend(dynamic_flags)
        except SandboxInfrastructureError:
            raise
        except Exception as ex:
            flags.append(f'SANDBOX_EXEC_ERROR: {type(ex).__name__}: {ex}')

        if flags:
            all_issues.append((i, seed, par, flags))

    # ── report ─────────────────────────────────────────────────────────────
    bad = len(all_issues)
    clean = total - bad
    print(f'Audited {total} entries  |  clean: {clean}  |  issues: {bad}')
    print('=' * 70)
    for entry_num, seed, par, flags in all_issues:
        print(f'\nEntry {entry_num} | seed={seed} | {par}')
        for flag in flags:
            print(f'  -> {flag}')

    if not all_issues:
        print('\nAll entries passed audit.')


if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else 'data/approved_curriculum.jsonl'
    audit(path)
