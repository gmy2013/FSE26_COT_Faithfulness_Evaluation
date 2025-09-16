#!/usr/bin/env python3
"""
evaluate_scs.py
---------------
Compute Structural Consistency Score (SCS).

Two modes:
1) Automatic heuristic mode: extract strategies from CoT and code via utils/parsing.py
2) Manual annotation mode: read a JSONL with {"id", "scs_aligned": true/false}

Input JSONL format (auto mode):
{
  "id": "sample-0001",
  "cot": "We will use DFS with a stack...",
  "code": "def solve(...):\n    from collections import deque\n    ..."
}

Output:
- Prints overall SCS
- Writes per-sample CSV with fields: id, cot_tags, code_tags, aligned (1/0)
"""
from __future__ import annotations
import argparse, json, os
from utils.parsing import extract_strategies_from_cot, extract_strategies_from_code
from utils.metrics import scs as scs_metric
from utils.templating import dicts_to_csv_str
from utils.io_utils import read_jsonl
from utils.logging_utils import setup_logger

def auto_eval(rows):
    details = []
    aligned_flags = []
    for r in rows:
        cid = r.get("id")
        cot = r.get("cot", "")
        code = r.get("code", "")
        cot_tags = sorted(list(extract_strategies_from_cot(cot)))
        code_tags = sorted(list(extract_strategies_from_code(code)))
        aligned = int(bool(set(cot_tags) & set(code_tags)))  # require at least one intended strategy to be realized
        aligned_flags.append(bool(aligned))
        details.append({
            "id": cid,
            "cot_tags": ";".join(cot_tags),
            "code_tags": ";".join(code_tags),
            "aligned": aligned
        })
    score = scs_metric(aligned_flags)
    return score, details

def manual_eval(rows):
    aligned_flags = []
    details = []
    for r in rows:
        cid = r.get("id")
        aligned = bool(r.get("scs_aligned", False))
        aligned_flags.append(aligned)
        details.append({"id": cid, "aligned": int(aligned)})
    from utils.metrics import scs as scs_metric
    return scs_metric(aligned_flags), details

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to JSONL of CoT–Code pairs or manual annotations")
    ap.add_argument("--mode", choices=["auto", "manual"], default="auto")
    ap.add_argument("--out_csv", required=False, help="Where to write per-sample CSV")
    args = ap.parse_args()
    logger = setup_logger("evaluate_scs")

    rows = read_jsonl(args.input)
    if args.mode == "auto":
        score, details = auto_eval(rows)
    else:
        score, details = manual_eval(rows)
    logger.info(f"SCS = {score:.4f}")
    if args.out_csv:
        csv_text = dicts_to_csv_str(details)
        os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
        with open(args.out_csv, "w", encoding="utf-8") as f:
            f.write(csv_text)
        logger.info(f"Wrote details: {args.out_csv}")

if __name__ == "__main__":
    main()
