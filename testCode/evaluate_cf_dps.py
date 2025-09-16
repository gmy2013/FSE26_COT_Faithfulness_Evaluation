#!/usr/bin/env python3
"""
evaluate_cf_dps.py
------------------
Metric extraction for Complexity Faithfulness (CF) and Differential Performance Score (DPS).

Input JSONL format expects per-sample records like:
{
  "id": "task-001",
  "code": "<python source code with entry function>",
  "entry": "solve",                               # entry function name for testing
  "sizes": [64, 128, 256, 512],
  "gen_args_template": {"n": "{size}"},   # optional: kwargs template for entry()
  "declared": {"time": "O(n log n)", "space": "O(n)"},  # CoT-declared classes
  "gold_cost": {"runtime": 0.1, "mem": 5e6, "instr": 10000},            # optional
  "baseline_cost": {"runtime": 0.5, "mem": 20e6, "instr": 60000}        # optional
}

We will:
- For each size s, build args/kwargs, run the entry and record runtime, peak mem, approx instr.
- Fit empirical complexity class from (sizes, runtimes) and (sizes, memory) to compare with declared.
- Compute CF (Eq. CF).
- If gold/baseline costs are provided, we aggregate measured costs at max size into model_cost and compute DPS.
  (Else DPS is omitted for that sample.)

Outputs:
- Prints dataset-level CF and average DPS (over samples where DPS was computable).
- Writes per-sample JSONL with measured sequences and inferred classes if --out is given.
"""
from __future__ import annotations
import argparse, json, os
from typing import Dict, Any, List
from utils.runtime_profile import run_function_in_subprocess
from utils.metrics import infer_complexity_class, cf as cf_metric, dps as dps_metric
from utils.io_utils import read_jsonl, write_jsonl
from utils.logging_utils import setup_logger

def _render_kwargs(template: Dict[str, str], size_val: int) -> Dict[str, Any]:
    # Replace "{size}" placeholders with concrete values
    out = {}
    for k, v in (template or {}).items():
        if isinstance(v, str):
            out[k] = v.replace("{size}", str(size_val))
            try:
                if "." in out[k]:
                    out[k] = float(out[k])
                else:
                    out[k] = int(out[k])
            except Exception:
                pass
        else:
            out[k] = v
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="JSONL of tasks")
    ap.add_argument("--repeat", type=int, default=1, help="Repetitions per measurement size")
    ap.add_argument("--timeout", type=float, default=30.0, help="Per-run timeout (s)")
    ap.add_argument("--out", help="Write enriched per-sample JSONL here")
    args = ap.parse_args()
    logger = setup_logger("evaluate_cf_dps")

    rows = read_jsonl(args.input)
    enriched = []
    declared_list = []
    sizes_list = []
    time_list = []
    space_list = []
    dps_scores = []

    for r in rows:
        code = r["code"]
        entry = r["entry"]
        sizes = r.get("sizes", [])
        gen_tpl = r.get("gen_args_template", {})
        declared = r.get("declared", {})
        gold_cost = r.get("gold_cost")
        baseline_cost = r.get("baseline_cost")

        runtimes = []
        mems = []
        instrs = []

        for s in sizes:
            kwargs = _render_kwargs(gen_tpl, s) if gen_tpl else {"n": s}
            res = run_function_in_subprocess(code, entry, args=[], kwargs=kwargs, repeat=args.repeat, timeout=args.timeout)
            if not res.get("ok", False):
                logger.warning(f"[{r.get('id')}] run failed at size={s}: {res.get('error')}")
                runtimes.append(float("inf"))
                mems.append(float("inf"))
                instrs.append(float("inf"))
            else:
                runtimes.append(res["runtime_s"] or 0.0)
                mems.append((res["peak_bytes"] or 0))
                instrs.append((res["approx_instr"] or 0))

        emp_time_class = infer_complexity_class(sizes, runtimes) if all(x and x!=float('inf') for x in runtimes) else None
        emp_space_class = infer_complexity_class(sizes, mems) if all(x and x!=float('inf') for x in mems) else None

        enriched.append({
            **r,
            "measured": {
                "runtimes_s": runtimes,
                "peak_bytes": mems,
                "approx_instr": instrs,
            },
            "empirical": {
                "time": emp_time_class,
                "space": emp_space_class,
            }
        })

        declared_list.append(declared or {})
        sizes_list.append(sizes)
        time_list.append(runtimes)
        space_list.append(mems)

        if gold_cost and baseline_cost:
            model_cost = {
                "runtime": (runtimes[-1] if runtimes and runtimes[-1] != float('inf') else float('inf')),
                "mem": (mems[-1] if mems and mems[-1] != float('inf') else float('inf')),
                "instr": (instrs[-1] if instrs and instrs[-1] != float('inf') else float('inf'))
            }
            dps_val = dps_metric(model_cost, gold_cost, baseline_cost)
            dps_scores.append(dps_val)

    cf_val = cf_metric(declared_list, sizes_list, time_list, space_list)

    logger.info(f"CF = {cf_val:.4f}")
    if dps_scores:
        logger.info(f"DPS (mean over computable samples) = {sum(dps_scores)/len(dps_scores):.4f}")
    else:
        logger.info("DPS (no gold/baseline costs provided across samples)")

    if args.out:
        write_jsonl(args.out, enriched)
        logger.info(f"Wrote enriched JSONL: {args.out}")

if __name__ == "__main__":
    main()
