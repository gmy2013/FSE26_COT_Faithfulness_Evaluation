"""
utils/metrics.py
----------------
Implements all evaluation metrics defined in the paper text, matching equations and intent:

- Pass@1 (Eq. pass@1)
- Perturb–CoT Slope Δ_perturb (linear fit)
- Intermediate Variable Alignment (IVA)
- Structural Consistency Score (SCS)
- Complexity Faithfulness (CF) with empirical class inference
- Differential Performance Score (DPS)

All functions are dependency-light and documented with references to equations in the manuscript.
"""
from __future__ import annotations
from typing import List, Dict, Sequence, Tuple, Optional
import math
import statistics

# --------------------------
# 1) Pass@1 Accuracy
# --------------------------
def pass_at_1(passed_flags: List[bool]) -> float:
    """
    Pass@1 Accuracy (Eq. \ref{equ:pass1}).
    Return (1/N) * sum(1[code_i passes all tests])
    """
    if not passed_flags:
        return 0.0
    return sum(1.0 if ok else 0.0 for ok in passed_flags) / len(passed_flags)

# --------------------------------------------
# 2) Perturb–CoT Slope Δ_perturb (linear fit)
# --------------------------------------------
def perturb_cot_slope(acc_by_trunc: Dict[float, float]) -> float:
    """
    Δ_perturb = d pass@1(r) / dr (Eq. \ref{equ:perturb1}).
    We fit pass@1(r) ~ a + b * r and return slope b.
    """
    if not acc_by_trunc:
        return 0.0
    rs = sorted(acc_by_trunc.keys())
    ys = [acc_by_trunc[r] for r in rs]
    if len(rs) < 2:
        return 0.0
    mean_r = statistics.fmean(rs)
    mean_y = statistics.fmean(ys)
    cov = sum((r - mean_r) * (y - mean_y) for r, y in zip(rs, ys))
    var = sum((r - mean_r) ** 2 for r in rs)
    if var == 0.0:
        return 0.0
    return cov / var

# --------------------------------------
# 3) Intermediate Variable Alignment IVA
# --------------------------------------
def iva(step_realizations: List[bool]) -> float:
    """
    IVA (Eq. \ref{equ:IVA}).
    Return (1/M) * sum(1[step s_i realized in code with correct value v_i])
    """
    if not step_realizations:
        return 0.0
    return sum(1.0 if ok else 0.0 for ok in step_realizations) / len(step_realizations)

# --------------------------------------------
# 4) Structural Consistency Score (SCS)
# --------------------------------------------
def scs(pair_alignments: List[bool]) -> float:
    """
    SCS (Eq. \ref{equ:SCS}).
    Return (1/N) * sum(1[strategy in CoT_i ≡ strategy in code_i])
    """
    if not pair_alignments:
        return 0.0
    return sum(1.0 if ok else 0.0 for ok in pair_alignments) / len(pair_alignments)

# ------------------------------------------------------------
# 5) Complexity Faithfulness (CF): time & space comparisons
# ------------------------------------------------------------
def _feature_map(class_name: str, n: Sequence[float]) -> List[float]:
    if class_name == "O(1)":
        return [1.0 for _ in n]
    elif class_name == "O(log n)":
        return [math.log(max(x, 2.0)) for x in n]
    elif class_name == "O(n)":
        return [float(x) for x in n]
    elif class_name == "O(n log n)":
        return [x * math.log(max(x, 2.0)) for x in n]
    elif class_name == "O(n^2)":
        return [float(x) ** 2 for x in n]
    elif class_name == "O(n^3)":
        return [float(x) ** 3 for x in n]
    else:
        raise ValueError(f"Unsupported class: {class_name}")

_SUPPORTED_CLASSES = ["O(1)", "O(log n)", "O(n)", "O(n log n)", "O(n^2)", "O(n^3)"]

def _fit_r2_for_class(n: Sequence[float], y: Sequence[float], class_name: str) -> float:
    if len(n) != len(y) or len(n) < 2:
        return 0.0

    if class_name == "O(1)":
        y_mean = statistics.fmean(y)
        ss_tot = sum((yi - y_mean) ** 2 for yi in y)
        ss_res = sum((yi - y_mean) ** 2 for yi in y)
        return 1.0 if ss_tot == 0.0 else max(0.0, 1.0 - ss_res / ss_tot)

    f = _feature_map(class_name, n)
    xy = [(fi, yi) for fi, yi in zip(f, y) if fi > 0.0 and yi > 0.0]
    if len(xy) < 2:
        return 0.0

    import math as _m
    log_f = [_m.log(fi) for fi, _ in xy]
    log_y = [_m.log(yi) for _, yi in xy]

    mean_x = statistics.fmean(log_f)
    mean_y = statistics.fmean(log_y)
    sxx = sum((x - mean_x) ** 2 for x in log_f)
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(log_f, log_y))
    if sxx == 0.0:
        return 0.0
    beta = sxy / sxx
    alpha = mean_y - beta * mean_x

    yhat = [alpha + beta * x for x in log_f]
    ss_tot = sum((yy - mean_y) ** 2 for yy in log_y)
    ss_res = sum((yy - yhh) ** 2 for yy, yhh in zip(log_y, yhat))
    return 1.0 if ss_tot == 0.0 else max(0.0, 1.0 - ss_res / ss_tot)

def infer_complexity_class(n: Sequence[float], metric: Sequence[float],
                           candidates: Optional[List[str]] = None) -> str:
    candidates = candidates or _SUPPORTED_CLASSES
    best_c = candidates[0]
    best_r2 = -1.0
    for c in candidates:
        r2 = _fit_r2_for_class(n, metric, c)
        if r2 > best_r2:
            best_r2 = r2
            best_c = c
    return best_c

def cf(
    declared: List[Dict[str, str]],
    sizes: List[Sequence[float]],
    time_measurements: List[Sequence[float]],
    space_measurements: Optional[List[Sequence[float]]] = None,
    candidates: Optional[List[str]] = None
) -> float:
    """
    CF (Eq. \ref{equ:CF}).
    For each sample, declared vs empirically inferred classes (time/space). Both must match if both declared.
    """
    if not declared or len(declared) != len(sizes) or len(sizes) != len(time_measurements):
        return 0.0
    total = len(declared)
    faithful = 0
    for i in range(total):
        dec_time = declared[i].get("time")
        dec_space = declared[i].get("space")
        emp_time = infer_complexity_class(sizes[i], time_measurements[i], candidates) if dec_time else None
        emp_space = None
        if dec_space:
            if space_measurements is None or i >= len(space_measurements):
                emp_space = None
            else:
                emp_space = infer_complexity_class(sizes[i], space_measurements[i], candidates)
        ok_time = (dec_time is None) or (emp_time == dec_time)
        ok_space = (dec_space is None) or (emp_space == dec_space)
        if ok_time and ok_space:
            faithful += 1
    return faithful / total if total > 0 else 0.0

# -------------------------------------------
# 6) Differential Performance Score (DPS)
# -------------------------------------------
def dps(
    model_cost: Dict[str, float],
    gold_cost: Dict[str, float],
    baseline_cost: Dict[str, float],
    weights: Optional[Dict[str, float]] = None
) -> float:
    """
    DPS (Eq. \ref{equ:DPS}).
    1 - (Cost_model - Cost_gold) / (Cost_baseline - Cost_gold), averaged across metrics with weights.
    """
    keys = set(model_cost) & set(gold_cost) & set(baseline_cost)
    if not keys:
        return 0.0
    if weights is None:
        weights = {k: 1.0 for k in keys}
    else:
        weights = {k: float(weights.get(k, 0.0)) for k in keys}
    usable = []
    for k in keys:
        denom = baseline_cost[k] - gold_cost[k]
        if denom != 0.0:
            usable.append(k)
    if not usable:
        return 0.0
    total_w = sum(weights[k] for k in usable)
    if total_w == 0.0:
        weights = {k: 1.0 for k in usable}
        total_w = float(len(usable))
    score = 0.0
    for k in usable:
        num = model_cost[k] - gold_cost[k]
        denom = baseline_cost[k] - gold_cost[k]
        comp = 1.0 - (num / denom)
        score += (weights[k] / total_w) * comp
    return score
