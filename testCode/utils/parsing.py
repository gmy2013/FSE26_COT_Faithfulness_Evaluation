"""
utils/parsing.py
----------------
Parsing and heuristic extraction utilities for:
- Strategies/data-structures mentioned in CoT text
- Strategies/data-structures implemented in Python code (AST + regex heuristics)

Targets: recursion, dynamic programming (memoization/tabulation), DFS, BFS, greedy,
heap-based approaches, sorting variants, hashmap, stack/queue, tree/graph traversal.
"""
from __future__ import annotations
from typing import Set, Dict, Any, Tuple, List
import re, ast

# --------- CoT text parsing ---------
COT_KEYWORDS = {
    "recursion": [r"\brecurs(ion|ive)\b", r"\bdfs\b", r"\bdepth[- ]?first\b"],
    "dp": [r"\bdp\b", r"\bdynamic programming\b", r"\bmemo(?:ization|ise)\b", r"\btabulation\b"],
    "dfs": [r"\bdfs\b", r"\bdepth[- ]?first\b"],
    "bfs": [r"\bbfs\b", r"\bbreadth[- ]?first\b"],
    "greedy": [r"\bgreedy\b"],
    "heap": [r"\bheap\b", r"\bpriority queue\b", r"\bmin-heap\b", r"\bmax-heap\b"],
    "sort": [r"\bsort(ing)?\b", r"\bheap sort\b", r"\bquick ?sort\b", r"\bmerge ?sort\b"],
    "hashmap": [r"\bhash(?:[- ]?)?map\b", r"\bdict(ionary)?\b", r"\bhashtable\b"],
    "stack": [r"\bstack\b"],
    "queue": [r"\bqueue\b", r"\bdeque\b"],
    "tree": [r"\btree\b", r"\bbinary tree\b", r"\bbst\b"],
    "graph": [r"\bgraph\b", r"\badjacency\b"],
}

def extract_strategies_from_cot(text: str) -> Set[str]:
    text_l = text.lower()
    found = set()
    for tag, pats in COT_KEYWORDS.items():
        for p in pats:
            if re.search(p, text_l):
                found.add(tag)
                break
    return found

# --------- Code analysis via AST ---------
class CodeVisitor(ast.NodeVisitor):
    def __init__(self):
        self.calls = set()
        self.names = set()
        self.attr = set()
        self.has_recursion = False
        self.uses_memo = False
        self.uses_tabulation = False
        self.uses_heap = False
        self.uses_stack = False
        self.uses_queue = False
        self.uses_dict = False
        self.uses_sort = False
        self.uses_graph = False
        self.uses_tree = False
        self.uses_dfs = False
        self.uses_bfs = False

        self._fn_stack: List[str] = []

    def visit_FunctionDef(self, node: ast.FunctionDef):
        self._fn_stack.append(node.name)
        self.generic_visit(node)
        self._fn_stack.pop()

    def visit_Call(self, node: ast.Call):
        fn = None
        if isinstance(node.func, ast.Name):
            fn = node.func.id
        elif isinstance(node.func, ast.Attribute):
            fn = node.func.attr
        if fn:
            self.calls.add(fn)

        if isinstance(node.func, ast.Attribute):
            if node.func.attr in ("lru_cache", "cache"):
                self.uses_memo = True

        if isinstance(node.func, ast.Attribute):
            if node.func.attr in ("heappush", "heappop", "heapify", "nlargest", "nsmallest"):
                self.uses_heap = True
        if isinstance(node.func, ast.Name) and node.func.id in ("heappush", "heappop", "heapify"):
            self.uses_heap = True

        if isinstance(node.func, ast.Name) and node.func.id in ("deque", "Queue"):
            self.uses_queue = True

        if isinstance(node.func, ast.Attribute) and node.func.attr in ("sort", "sorted"):
            self.uses_sort = True
        if isinstance(node.func, ast.Name) and node.func.id == "sorted":
            self.uses_sort = True

        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute):
        self.attr.add(node.attr)
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name):
        self.names.add(node.id)
        if self._fn_stack and node.id == self._fn_stack[-1]:
            self.has_recursion = True
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign):
        if isinstance(node.value, (ast.List, ast.ListComp)):
            self.uses_tabulation = True
        if isinstance(node.value, ast.Dict):
            self.uses_dict = True
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript):
        self.uses_tabulation = True
        self.generic_visit(node)

def extract_strategies_from_code(code: str) -> Set[str]:
    tags = set()
    try:
        tree = ast.parse(code)
    except Exception:
        code_l = code.lower()
        if "heapq" in code_l: tags.add("heap")
        if "def " in code_l and re.search(r"\breturn\b.*\b\w+\(", code_l):
            tags.add("recursion")
        if "lru_cache" in code_l: tags.add("dp")
        if "sorted(" in code_l or ".sort(" in code_l: tags.add("sort")
        if "dict(" in code_l or "{" in code_l: tags.add("hashmap")
        if "deque(" in code_l: tags.add("queue")
        return tags

    v = CodeVisitor()
    v.visit(tree)

    if v.has_recursion: tags.add("recursion")
    if v.uses_memo or v.uses_tabulation: tags.add("dp")
    if v.uses_heap or ("heapq" in v.names) or ("heapq" in v.attr): tags.add("heap")
    if v.uses_sort: tags.add("sort")
    if v.uses_dict or ("dict" in v.names): tags.add("hashmap")
    if v.uses_queue or ("deque" in v.names): tags.add("queue")
    if "dfs" in v.names or "dfs" in v.calls: tags.add("dfs")
    if "bfs" in v.names or "bfs" in v.calls: tags.add("bfs")

    code_l = code.lower()
    if re.search(r"class\s+\w*node\b", code_l) or "left" in code_l and "right" in code_l:
        tags.add("tree")
    if "adjacency" in code_l or "graph" in code_l:
        tags.add("graph")

    return tags
