import ast
import builtins
import hashlib
from typing import Any, Dict, List, Set, Tuple

BUILTIN_IDENTIFIERS = set(dir(builtins))


class ASTCanonicalizer(ast.NodeTransformer):
    """Normalizes variable names, docstrings, function names, classes, async functions, and call keywords."""

    def __init__(self):
        self.mapping = {}
        self.var_idx = 0
        self.func_idx = 0

    def _get_var(self, name: str) -> str:
        if name in BUILTIN_IDENTIFIERS:
            return name
        if name not in self.mapping:
            self.mapping[name] = f"v_{self.var_idx}"
            self.var_idx += 1
        return self.mapping[name]

    def visit_Name(self, node: ast.Name) -> ast.Name:
        node.id = self._get_var(node.id)
        return self.generic_visit(node)

    def visit_arg(self, node: ast.arg) -> ast.arg:
        node.arg = self._get_var(node.arg)
        return self.generic_visit(node)

    def visit_keyword(self, node: ast.keyword) -> ast.keyword:
        if node.arg:
            node.arg = self._get_var(node.arg)
        return self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        self._strip_docstring(node)
        if node.name not in self.mapping:
            self.mapping[node.name] = f"fn_{self.func_idx}"
            self.func_idx += 1
        node.name = self.mapping[node.name]
        return self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AsyncFunctionDef:
        self._strip_docstring(node)
        if node.name not in self.mapping:
            self.mapping[node.name] = f"fn_{self.func_idx}"
            self.func_idx += 1
        node.name = self.mapping[node.name]
        return self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.ClassDef:
        self._strip_docstring(node)
        if node.name not in self.mapping:
            self.mapping[node.name] = f"cls_{self.func_idx}"
            self.func_idx += 1
        node.name = self.mapping[node.name]
        return self.generic_visit(node)

    def _strip_docstring(self, node: Any) -> None:
        if (
            node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)
        ):
            node.body.pop(0)
            if not node.body:
                node.body.append(ast.Pass())


def get_canonical_tokens(code_str: str) -> List[str]:
    """Parses Python code and yields normalized AST token streams."""
    try:
        tree = ast.parse(code_str)
        norm_tree = ASTCanonicalizer().visit(tree)
        ast.fix_missing_locations(norm_tree)
        dumped = ast.dump(norm_tree, annotate_fields=False, include_attributes=False)
        return dumped.split()
    except Exception:
        return code_str.split()


def get_ngrams(tokens: List[str], k: int = 4) -> Set[Tuple[str, ...]]:
    """Extracts overlapping n-gram tuples of size k from ordered token stream."""
    if not tokens:
        return set()
    eff_k = min(k, len(tokens))
    return {tuple(tokens[i : i + eff_k]) for i in range(len(tokens) - eff_k + 1)}


def compute_jaccard_similarity(tokens_a: List[str], tokens_b: List[str], k: int = 4) -> float:
    """Calculates Jaccard similarity index over adaptive n-grams of ordered AST token sequences."""
    if not tokens_a or not tokens_b:
        return 0.0
    eff_k = max(1, min(k, len(tokens_a), len(tokens_b)))
    ngrams_a = get_ngrams(tokens_a, k=eff_k)
    ngrams_b = get_ngrams(tokens_b, k=eff_k)
    if not ngrams_a or not ngrams_b:
        return 0.0
    intersection = len(ngrams_a.intersection(ngrams_b))
    union = len(ngrams_a.union(ngrams_b))
    return intersection / union


class CorpusBalanceEngine:
    """Manages MinHash LSH deduplication with AST n-gram topology and paradigm balancing caps."""

    def __init__(self, max_per_paradigm: int = 35, similarity_threshold: float = 0.70, ngram_k: int = 4):
        self.max_per_paradigm = max_per_paradigm
        self.similarity_threshold = similarity_threshold
        self.ngram_k = ngram_k
        self.paradigm_counts: Dict[str, int] = {}
        self.corpus_tokens: List[List[str]] = []

    def can_accept_paradigm(self, paradigm: str) -> bool:
        return self.paradigm_counts.get(paradigm, 0) < self.max_per_paradigm

    def is_duplicate(self, code_str: str) -> bool:
        candidate_tokens = get_canonical_tokens(code_str)
        for existing_tokens in self.corpus_tokens:
            sim = compute_jaccard_similarity(candidate_tokens, existing_tokens, k=self.ngram_k)
            if sim >= self.similarity_threshold:
                return True
        return False

    def register_candidate(self, paradigm: str, code_str: str) -> None:
        tokens = get_canonical_tokens(code_str)
        self.corpus_tokens.append(tokens)
        self.paradigm_counts[paradigm] = self.paradigm_counts.get(paradigm, 0) + 1
