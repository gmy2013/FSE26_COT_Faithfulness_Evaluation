#!/usr/bin/env python3
"""
logic_repair_sat.py
-------------------
Constraint-based CoT repair via a built-in SAT/MaxSAT solver (no external deps).

Goal:
- Given mismatches between CoT-declared strategies and code-implemented strategies,
  propose a minimal set of CoT edits (add/remove tags) so that constraints hold.

Problem modeling:
- For each tag t in TAGS, a boolean variable COT_t denotes "CoT declares tag t".
- We have observed CODE_t (fixed booleans) from code analysis.
- Constraints:
    1) If CODE_heap is True then (COT_heap is True)    # must acknowledge induced/used strategy
    2) If COT_dp is True then not (COT_greedy and COT_heap) simultaneously  # example of mutual-exclusion (optional)
    3) User-provided hard clauses in CNF over COT_t literals.
- Soft objective: minimize number of flips from the current COT assignment.

Input JSON:
{
  "cot_tags": ["dfs"],
  "code_tags": ["dfs", "heap"],
  "hard_clauses": [["COT_heap"], ["-COT_greedy", "COT_dp"]]
}

Output:
- Repaired CoT tag set (closest in Hamming distance) that satisfies all hard constraints.
- A diff of edits (added/removed tags).
"""
from __future__ import annotations
import argparse, json, itertools
from typing import List, Dict, Set, Tuple

TAGS = ["recursion","dp","dfs","bfs","greedy","heap","sort","hashmap","stack","queue","tree","graph"]

def cnf_eval(assignment: Dict[str, bool], clauses: List[List[str]]) -> bool:
    """Evaluate CNF under assignment. Literal format: 'COT_heap' or '-COT_heap'."""
    def lit_value(lit: str) -> bool:
        neg = lit.startswith("-")
        var = lit[1:] if neg else lit
        val = assignment.get(var, False)
        return (not val) if neg else val
    for clause in clauses:
        if not any(lit_value(l) for l in clause):
            return False
    return True

def build_hard_clauses(code_tags: Set[str], user_clauses: List[List[str]]) -> List[List[str]]:
    clauses = []
    for t in code_tags:
        clauses.append([f"COT_{t}"])
    clauses.append(["-COT_dp", "-COT_greedy", "-COT_heap"])
    for cl in user_clauses or []:
        clauses.append(cl)
    return clauses

def initial_assignment(cot_tags: Set[str]) -> Dict[str, bool]:
    return {f"COT_{t}": (t in cot_tags) for t in TAGS}

def hamming(a: Dict[str, bool], b: Dict[str, bool]) -> int:
    return sum(1 for k in a if a[k] != b.get(k, False))

def neighbors(assign: Dict[str, bool]) -> List[Dict[str, bool]]:
    outs = []
    for k in assign:
        n = assign.copy()
        n[k] = not n[k]
        outs.append(n)
    return outs

def dpll_sat(clauses: List[List[str]], vars_list: List[str], assignment: Dict[str, bool]) -> Tuple[bool, Dict[str, bool]]:
    """Simple DPLL SAT (for small var counts). Returns (satisfiable, model)."""
    changed = True
    while changed:
        changed = False
        for clause in clauses:
            unassigned = []
            satisfied = False
            for lit in clause:
                neg = lit.startswith("-")
                var = lit[1:] if neg else lit
                if var in assignment:
                    val = assignment[var]
                    if (not val if neg else val):
                        satisfied = True
                        break
                else:
                    unassigned.append(lit)
            if satisfied:
                continue
            if not unassigned:
                return False, {}
            if len(unassigned) == 1:
                l = unassigned[0]
                neg = l.startswith("-")
                var = l[1:] if neg else l
                assignment[var] = (not neg)
                changed = True
    vars_set = set(vars_list)
    unassigned_vars = [v for v in vars_list if v not in assignment]
    if not unassigned_vars:
        if cnf_eval(assignment, clauses):
            return True, assignment
        return False, {}
    v = unassigned_vars[0]
    for val in [True, False]:
        a2 = assignment.copy()
        a2[v] = val
        ok, model = dpll_sat(clauses, vars_list, a2)
        if ok:
            return True, model
    return False, {}

def maxsat_min_flips(initial: Dict[str, bool], clauses: List[List[str]]) -> Dict[str, bool]:
    """
    Find a satisfying assignment with minimum Hamming distance to initial.
    Brute-force search by increasing radius; uses DPLL per candidate by adding
    cardinality constraints as "at most k flips".
    For small variable counts this is fine.
    """
    vars_list = list(initial.keys())
    for k in range(0, len(vars_list) + 1):
        for flip_idxs in itertools.combinations(range(len(vars_list)), k):
            cand = {v: initial[v] for v in vars_list}
            for idx in flip_idxs:
                v = vars_list[idx]
                cand[v] = (not cand[v])
            ok, model = dpll_sat(clauses, vars_list, cand.copy())
            if ok:
                return model
    return initial

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="JSON specifying cot_tags, code_tags, hard_clauses")
    ap.add_argument("--out", required=False, help="Where to write result JSON")
    args = ap.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    cot_tags = set(cfg.get("cot_tags", []))
    code_tags = set(cfg.get("code_tags", []))
    user_clauses = cfg.get("hard_clauses", [])

    clauses = build_hard_clauses(code_tags, user_clauses)
    init = initial_assignment(cot_tags)
    model = maxsat_min_flips(init, clauses)

    repaired = sorted([t for t in TAGS if model.get(f"COT_{t}", False)])
    added = sorted(list(set(repaired) - cot_tags))
    removed = sorted(list(cot_tags - set(repaired)))

    out = {
        "repaired_cot_tags": repaired,
        "added": added,
        "removed": removed,
        "satisfied": cnf_eval(model, clauses),
        "hard_clauses": clauses
    }
    s = json.dumps(out, ensure_ascii=False, indent=2)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(s)
    print(s)

if __name__ == "__main__":
    main()
