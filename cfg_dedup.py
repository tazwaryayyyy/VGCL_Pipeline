"""cfg_dedup.py

Structural Deduplication Engine using AST Normalization and Control Flow Graph (CFG) Hashing.

Replaces token similarity with AST tree structure normalization, literal/identifier striping,
CFG block/edge graph extraction, and AST edit-distance structural scoring.
"""

import ast
import builtins
import hashlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

BUILTIN_NAMES = set(dir(builtins))


class ASTStructuralNormalizer(ast.NodeTransformer):
    """Scope-aware AST Normalizer that replaces variable/function names and literals with canonical placeholders."""

    def __init__(self):
        self.var_map: Dict[str, str] = {}
        self.var_count = 0

    def _get_canon_name(self, name: str) -> str:
        if name in BUILTIN_NAMES or name in {"solve", "solve_brute", "generate_test_cases"}:
            return name
        if name not in self.var_map:
            self.var_map[name] = f"var_{self.var_count}"
            self.var_count += 1
        return self.var_map[name]

    def visit_Name(self, node: ast.Name) -> ast.Name:
        node.id = self._get_canon_name(node.id)
        return self.generic_visit(node)

    def visit_arg(self, node: ast.arg) -> ast.arg:
        node.arg = self._get_canon_name(node.arg)
        return self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> ast.Constant:
        # Literal normalization: replace arbitrary numeric/string constants with type indicators
        if isinstance(node.value, bool):
            node.value = True
        elif isinstance(node.value, (int, float)):
            node.value = 0
        elif isinstance(node.value, str):
            node.value = "STR_LIT"
        return node


class CFGExtractor(ast.NodeVisitor):
    """Extracts Control Flow Graph (CFG) skeleton: basic blocks, loops, conditionals, and branch edges."""

    def __init__(self):
        self.nodes: List[str] = []
        self.edges: List[Tuple[str, str]] = []
        self.current_block = "ENTRY"
        self.block_counter = 0

    def _new_block(self, prefix: str) -> str:
        self.block_counter += 1
        name = f"{prefix}_{self.block_counter}"
        self.nodes.append(name)
        return name

    def visit_If(self, node: ast.If):
        header = self._new_block("IF_COND")
        self.edges.append((self.current_block, header))

        then_block = self._new_block("THEN")
        self.edges.append((header, then_block))

        prev = self.current_block
        self.current_block = then_block
        for stmt in node.body:
            self.visit(stmt)

        if node.orelse:
            else_block = self._new_block("ELSE")
            self.edges.append((header, else_block))
            self.current_block = else_block
            for stmt in node.orelse:
                self.visit(stmt)

        join_block = self._new_block("JOIN")
        self.edges.append((then_block, join_block))
        self.current_block = join_block

    def visit_For(self, node: ast.For):
        header = self._new_block("FOR_LOOP")
        self.edges.append((self.current_block, header))
        body_block = self._new_block("FOR_BODY")
        self.edges.append((header, body_block))

        self.current_block = body_block
        for stmt in node.body:
            self.visit(stmt)
        self.edges.append((self.current_block, header))  # Loop back edge

        exit_block = self._new_block("FOR_EXIT")
        self.edges.append((header, exit_block))
        self.current_block = exit_block

    def visit_While(self, node: ast.While):
        header = self._new_block("WHILE_LOOP")
        self.edges.append((self.current_block, header))
        body_block = self._new_block("WHILE_BODY")
        self.edges.append((header, body_block))

        self.current_block = body_block
        for stmt in node.body:
            self.visit(stmt)
        self.edges.append((self.current_block, header))  # Loop back edge

        exit_block = self._new_block("WHILE_EXIT")
        self.edges.append((header, exit_block))
        self.current_block = exit_block

    def visit_Return(self, node: ast.Return):
        ret_block = self._new_block("RETURN")
        self.edges.append((self.current_block, ret_block))
        self.current_block = ret_block


def compute_ast_hashes(code_str: str) -> Tuple[str, str, List[str]]:
    """Normalizes AST and computes AST hash and CFG graph hash."""
    try:
        tree = ast.parse(code_str)
        norm_tree = ASTStructuralNormalizer().visit(tree)
        ast.fix_missing_locations(norm_tree)
        dumped = ast.dump(norm_tree, annotate_fields=False, include_attributes=False)
        ast_hash = hashlib.sha256(dumped.encode("utf-8")).hexdigest()[:16]

        cfg = CFGExtractor()
        cfg.visit(norm_tree)
        cfg_str = f"NODES:{sorted(cfg.nodes)};EDGES:{sorted(cfg.edges)}"
        cfg_hash = hashlib.sha256(cfg_str.encode("utf-8")).hexdigest()[:16]

        tokens = dumped.split()
        return ast_hash, cfg_hash, tokens
    except Exception:
        fallback_hash = hashlib.sha256(code_str.encode("utf-8")).hexdigest()[:16]
        return fallback_hash, fallback_hash, code_str.split()


def compute_structural_similarity(tokens_a: List[str], tokens_b: List[str], k: int = 4) -> float:
    """Calculates adaptive n-gram Jaccard structural similarity over canonical AST tokens."""
    if not tokens_a or not tokens_b:
        return 0.0
    eff_k = max(1, min(k, len(tokens_a), len(tokens_b)))
    set_a = {tuple(tokens_a[i : i + eff_k]) for i in range(len(tokens_a) - eff_k + 1)}
    set_b = {tuple(tokens_b[i : i + eff_k]) for i in range(len(tokens_b) - eff_k + 1)}
    if not set_a or not set_b:
        return 0.0
    inter = len(set_a.intersection(set_b))
    union = len(set_a.union(set_b))
    return inter / union if union > 0 else 0.0


@dataclass
class DeduplicationResult:
    is_duplicate: bool
    ast_hash: str
    cfg_hash: str
    similarity_score: float
    nearest_neighbor_problem: Optional[str]


class StructuralDeduplicationEngine:
    """CFG and AST structural deduplication engine maintaining corpus tree state."""

    def __init__(self, similarity_threshold: float = 0.70):
        self.similarity_threshold = similarity_threshold
        self.corpus_problems: List[Dict[str, Any]] = []

    def check_candidate(self, problem_id: str, code_str: str) -> DeduplicationResult:
        ast_hash, cfg_hash, tokens = compute_ast_hashes(code_str)

        max_sim = 0.0
        nearest_id = None

        for existing in self.corpus_problems:
            if existing["cfg_hash"] == cfg_hash and existing["ast_hash"] == ast_hash:
                return DeduplicationResult(
                    is_duplicate=True,
                    ast_hash=ast_hash,
                    cfg_hash=cfg_hash,
                    similarity_score=1.0,
                    nearest_neighbor_problem=existing["problem_id"],
                )

            sim = compute_structural_similarity(tokens, existing["tokens"])
            if sim > max_sim:
                max_sim = sim
                nearest_id = existing["problem_id"]

        is_dup = max_sim >= self.similarity_threshold
        return DeduplicationResult(
            is_duplicate=is_dup,
            ast_hash=ast_hash,
            cfg_hash=cfg_hash,
            similarity_score=max_sim,
            nearest_neighbor_problem=nearest_id,
        )

    def register_candidate(self, problem_id: str, code_str: str) -> DeduplicationResult:
        res = self.check_candidate(problem_id, code_str)
        ast_hash, cfg_hash, tokens = compute_ast_hashes(code_str)
        self.corpus_problems.append({
            "problem_id": problem_id,
            "ast_hash": ast_hash,
            "cfg_hash": cfg_hash,
            "tokens": tokens,
        })
        return res


if __name__ == "__main__":
    engine = StructuralDeduplicationEngine(similarity_threshold=0.70)
    c1 = "def solve(a, b):\n    if a > b:\n        return a\n    return b"
    c2 = "def solve(x, y):\n    if x > y:\n        return x\n    return y"

    res1 = engine.register_candidate("P1", c1)
    res2 = engine.check_candidate("P2", c2)
    print("Code 1 reg:", res1)
    print("Code 2 check:", res2)
