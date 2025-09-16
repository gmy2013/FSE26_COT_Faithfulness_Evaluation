#!/usr/bin/env python3
"""
perturb_slope_analysis.py
-------------------------
Compute Perturb–CoT Slope (Δ_perturb) and Pass@1 under multiple truncation ratios.

Input JSONL format per sample:
{
  "id": "sample-42",
  "cot": "Step 1: ... Step 2: ...",               # full CoT
  "code_full": "<python code>",                   # code generated with full CoT (optional if code_by_trunc provided)
  "entry": "solve",                               # entry function name for testing
  "tests": { "command": "python tests/run_case.py --impl {code_file}" },  # how to test; non-zero exit => fail
  "truncation_grid": [0.2, 0.4, 0.6, 0.8, 1.0],   # retained proportions (default)
  "gen": { "command": "python generate_code.py --cot_file {cot_file} --out {code_file}" },
  "code_by_trunc": { "0.2": "<code at 20%>", ... }  # optional: if present, generation step is skipped
}

Behavior:
- For each ratio r, build a truncated CoT by taking the first ceil(r * steps) steps (split by lines/"Step").
- If code_by_trunc[r] exists, write it to a temp file; else call gen.command (with {cot_file},{code_file}).
- Run tests via tests.command with placeholders {code_file}; pass if exit code == 0.
- Aggregate pass@1(r) across samples and compute Δ_perturb via utils/metrics.perturb_cot_slope .
- Also compute overall pass@1 at each r across dataset.
"""
from __future__ import annotations
import argparse, os, sys, json, math, re, tempfile, subprocess
from typing import List, Dict
from utils.metrics import pass_at_1, perturb_cot_slope
from utils.io_utils import read_jsonl
from utils.logging_utils import setup_logger

def _split_steps(cot: str) -> List[str]:
    # Try to split by numbered "Step" or by lines as fallback
    pats = [r"(?:^|\n)\s*step\s*\d+\s*[:.\-]\s*", r"(?:^|\n)\s*步骤\s*\d+\s*[:.\-]\s*"]
    for p in pats:
        parts = re.split(p, cot, flags=re.IGNORECASE)
        if len(parts) > 1:
            # parts[0] is preamble; keep the rest
            return [s.strip() for s in parts[1:] if s.strip()]
    # fallback: line-based steps
    return [ln.strip() for ln in cot.splitlines() if ln.strip()]

def _truncate_cot(cot: str, ratio: float) -> str:
    steps = _split_steps(cot)
    if not steps:
        return cot if ratio >= 1.0 else ""
    k = max(1, math.ceil(len(steps) * ratio))
    return "\n".join(steps[:k])

def _write(tmpdir: str, name: str, content: str) -> str:
    path = os.path.join(tmpdir, name)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return path

def _run_cmd(cmd: str, timeout: float = 60.0) -> int:
    proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        return 124
    return proc.returncode

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="JSONL of samples")
    ap.add_argument("--timeout", type=float, default=60.0, help="Timeout for gen/test commands")
    args = ap.parse_args()
    logger = setup_logger("perturb_slope")

    rows = read_jsonl(args.input)
    acc_by_ratio: Dict[float, List[bool]] = {}

    for r in rows:
        cot = r.get("cot", "")
        entry = r.get("entry", "solve")
        grid = r.get("truncation_grid") or [0.25, 0.5, 0.75]
        gen = r.get("gen") or {}
        tests = r.get("tests") or {}
        code_by_trunc = r.get("code_by_trunc", {})

        with tempfile.TemporaryDirectory() as td:
            for ratio in grid:
                truncated = _truncate_cot(cot, float(ratio))
                cot_file = _write(td, f"cot_{ratio}.txt", truncated)
                code_file = os.path.join(td, f"impl_{ratio}.py")

                if str(ratio) in code_by_trunc:
                    _write(td, f"impl_{ratio}.py", code_by_trunc[str(ratio)])
                else:
                    gen_cmd_tpl = gen.get("command")
                    if not gen_cmd_tpl:
                        logger.warning(f"[{r.get('id')}] missing gen.command and no code_by_trunc for ratio={ratio}")
                        rc = 1
                    else:
                        cmd = gen_cmd_tpl.format(cot_file=cot_file, code_file=code_file)
                        rc = _run_cmd(cmd, timeout=args.timeout)
                        if rc != 0:
                            logger.warning(f"[{r.get('id')}] generation failed at ratio={ratio} (rc={rc})")

                test_cmd_tpl = tests.get("command")
                if not test_cmd_tpl:
                    logger.warning(f"[{r.get('id')}] no tests.command provided; marking as fail")
                    passed = False
                else:
                    tcmd = test_cmd_tpl.format(code_file=code_file)
                    trc = _run_cmd(tcmd, timeout=args.timeout)
                    passed = (trc == 0)

                acc_by_ratio.setdefault(float(ratio), []).append(bool(passed))

    acc_summary = {r: pass_at_1(flags) for r, flags in sorted(acc_by_ratio.items())}
    slope = perturb_cot_slope(acc_summary)
    logger.info("Pass@1 by ratio: " + ", ".join(f"{r:.2f}→{acc:.4f}" for r, acc in sorted(acc_summary.items())))
    logger.info(f"Δ_perturb (slope) = {slope:.6f}")

if __name__ == "__main__":
    main()
