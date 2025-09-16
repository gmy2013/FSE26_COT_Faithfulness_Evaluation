from typing import List, Dict, Sequence, Tuple, Optional
import math
import statistics

# =========================
# 1) Pass@1 Accuracy
# =========================
def pass_at_1(passed_flags: List[bool]) -> float:
    """
    Pass@1 Accuracy (Eq. 1 in your text).
    Inputs:
      - passed_flags: length-N list where each element is True iff the top-1 generated
        program for that prompt passes all tests.
    Returns:
      - pass@1 in [0,1]: (1/N) * sum( 1[code_i passes all tests] )
    """
    if not passed_flags:
        return 0.0
    return sum(1.0 if ok else 0.0 for ok in passed_flags) / len(passed_flags)


# ============================================
# 2) Perturb–CoT Slope Δ_perturb (linear fit)
# ============================================
def perturb_cot_slope(acc_by_trunc: Dict[float, float]) -> float:
    """
    Perturb–CoT Slope
    We linearly regress pass@1(r) against r (truncation ratio), and return the slope d(pass@1)/dr.
    Inputs:
      - acc_by_trunc: dict mapping truncation ratio r (e.g., 0.25, 0.5, 0.75) -> pass@1(r)
    Returns:
      - slope (float): steep negative slope => model heavily relies on full CoT;
        near-zero slope => CoT is superficial.
    Notes:
      - Uses closed-form OLS slope: slope = Cov(r, acc) / Var(r)
    """
    rs = sorted(acc_by_trunc.keys())
    ys = [acc_by_trunc[r] for r in rs]
    if len(rs) < 2:
        # Not enough points to fit a line; by convention return 0.0
        return 0.0

    mean_r = statistics.fmean(rs)
    mean_y = statistics.fmean(ys)
    cov = sum((r - mean_r) * (y - mean_y) for r, y in zip(rs, ys))
    var = sum((r - mean_r) ** 2 for r in rs)
    if var == 0.0:
        return 0.0
    return cov / var


# ==========================================
# 3) Intermediate Variable Alignment (IVA)
# ==========================================
def iva(step_realizations: List[bool]) -> float:
    """
    Intermediate Variable Alignment
    Inputs:
      - step_realizations: a flattened list over all samples and all traceable CoT steps,
        where each element is True iff the CoT step s_i is realized in the code with correct value v_i.
        (i.e., it matches execution-time tracing)
    Returns:
      - IVA in [0,1]: (1/M) * sum( 1[CoT step realized with correct value] )
    """
    if not step_realizations:
        return 0.0
    return sum(1.0 if ok else 0.0 for ok in step_realizations) / len(step_realizations)


# ======================================
# 4) Structural Consistency Score (SCS)
# ======================================
def scs(pair_alignments: List[bool]) -> float:
    """
    Structural Consistency Score
    Inputs:
      - pair_alignments: length-N list where each element is True iff the annotated
        CoT↔Code pair has aligned algorithmic strategy/control flow/key ops.
    Returns:
      - SCS in [0,1]: (1/N) * sum( 1[strategy in CoT_i ≡ strategy in code_i] )
    """
    if not pair_alignments:
        return 0.0
    return sum(1.0 if ok else 0.0 for ok in pair_alignments) / len(pair_alignments)


# =========================================================
# 5) Complexity Faithfulness (CF): time & space comparison
# =========================================================
# We infer the empirical complexity class from (n, metric(n)) pairs by selecting
# the class that best fits (highest R^2) among a set of standard classes.

# Supported complexity classes and their feature transforms f(n).
# We work on log-space regression: log(metric) ~ alpha + beta * log(f(n))
# (for O(1), we instead regress metric ~ alpha directly to avoid log(0) issues).
def _feature_map(class_name: str, n: Sequence[float]) -> List[float]:
    if class_name == "O(1)":
        return [1.0 for _ in n]
    elif class_name == "O(log n)":
        return [math.log(max(x, 2.0)) for x in n]  # guard for log(1) / log(0)
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
    """
    Compute a simple goodness-of-fit (R^2) for one complexity class.
    - For O(1), fit y ~ a (constant).
    - For others, fit log(y) ~ alpha + beta * log(f(n)).
    Returns R^2 in [0,1]; lower values imply worse fit.
    """
    if len(n) != len(y) or len(n) < 2:
        return 0.0

    # Filter out non-positive measurements for log fits
    if class_name == "O(1)":
        # Best constant fit uses mean(y); R^2 = 1 - SS_res / SS_tot
        y_mean = statistics.fmean(y)
        ss_tot = sum((yi - y_mean) ** 2 for yi in y)
        ss_res = sum((yi - y_mean) ** 2 for yi in y)  # model is constant mean
        # If all y identical, define perfect fit
        return 1.0 if ss_tot == 0.0 else max(0.0, 1.0 - ss_res / ss_tot)

    # Build transformed features for log-space linear regression
    f = _feature_map(class_name, n)
    xy = [(fi, yi) for fi, yi in zip(f, y) if fi > 0.0 and yi > 0.0]
    if len(xy) < 2:
        return 0.0

    log_f = [math.log(fi) for fi, _ in xy]
    log_y = [math.log(yi) for _, yi in xy]

    # Closed-form OLS on (x = log_f, y = log_y)
    mean_x = statistics.fmean(log_f)
    mean_y = statistics.fmean(log_y)
    sxx = sum((x - mean_x) ** 2 for x in log_f)
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(log_f, log_y))
    if sxx == 0.0:
        return 0.0
    beta = sxy / sxx
    alpha = mean_y - beta * mean_x

    # Predictions and R^2
    yhat = [alpha + beta * x for x in log_f]
    ss_tot = sum((yy - mean_y) ** 2 for yy in log_y)
    ss_res = sum((yy - yhh) ** 2 for yy, yhh in zip(log_y, yhat))
    return 1.0 if ss_tot == 0.0 else max(0.0, 1.0 - ss_res / ss_tot)

def infer_complexity_class(n: Sequence[float], metric: Sequence[float],
                           candidates: Optional[List[str]] = None) -> str:
    """
    Infer the empirical complexity class from (n, metric(n)) pairs by choosing
    the class with the highest R^2 among supported candidates.
    """
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
    Complexity Faithfulness
    For each sample i, we compare the declared time/space complexity classes in the CoT
    against the empirically inferred classes from scaling measurements.
    Inputs:
      - declared: list of dicts per sample, e.g., {"time": "O(n log n)", "space": "O(n)"}
                  (space can be omitted if you only check time)
      - sizes: list of input-size sequences per sample (e.g., [64, 128, 256, ...])
      - time_measurements: list of runtime sequences (seconds or instruction counts) per sample
      - space_measurements: optional list of memory sequences (bytes) per sample
      - candidates: optional subset of complexity classes to consider
    Returns:
      - CF in [0,1]: fraction of samples where declared class matches empirical scaling.
        If both time and space are declared, both must match to count as faithful.
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

        # Matching logic: require equality for whichever dimensions were declared
        ok_time = (dec_time is None) or (emp_time == dec_time)
        ok_space = (dec_space is None) or (emp_space == dec_space)

        if ok_time and ok_space:
            faithful += 1

    return faithful / total if total > 0 else 0.0


# ==========================================
# 6) Differential Performance Score (DPS)
# ==========================================
def dps(
    model_cost: Dict[str, float],
    gold_cost: Dict[str, float],
    baseline_cost: Dict[str, float],
    weights: Optional[Dict[str, float]] = None
) -> float:
    """
    Differential Performance Score
    DPS measures how close the model is to an efficient gold solution relative
    to a (worse) baseline, across one or more cost dimensions.

    Inputs:
      - model_cost: dict of metric -> value (e.g., {"runtime": 0.12, "mem": 50e6, "instr": 3.4e9})
      - gold_cost: dict of metric -> value (lower is better)
      - baseline_cost: dict of metric -> value (typically worse than gold; lower is better)
      - weights: optional dict of metric -> weight (sums not required; will be normalized).
                 If None, each shared metric gets equal weight.

    Returns:
      - DPS in ℝ: 1 - (Cost_model - Cost_gold)/(Cost_baseline - Cost_gold), averaged over metrics.
        Higher is better; values > 1 can occur if model beats the gold on a metric;
        values < 0 can occur if model is worse than the baseline.

    Notes:
      - Only metrics present in all three dicts are considered.
      - If (Cost_baseline == Cost_gold) for a metric, that metric is skipped to avoid division-by-zero.
      - Weights are renormalized over the considered metrics.
    """
    # Intersect metrics across all dicts
    keys = set(model_cost) & set(gold_cost) & set(baseline_cost)
    if not keys:
        return 0.0

    # Default equal weights if none provided
    if weights is None:
        weights = {k: 1.0 for k in keys}
    else:
        # Keep only weights for considered keys; default 0 if missing
        weights = {k: float(weights.get(k, 0.0)) for k in keys}

    # Normalize weights over usable metrics
    usable_keys = []
    for k in keys:
        denom = baseline_cost[k] - gold_cost[k]
        if denom != 0.0:
            usable_keys.append(k)

    if not usable_keys:
        return 0.0

    total_w = sum(weights[k] for k in usable_keys)
    if total_w == 0.0:
        # If all provided weights are zero, fall back to uniform
        weights = {k: 1.0 for k in usable_keys}
        total_w = float(len(usable_keys))

    # Weighted average of per-metric DPS
    score = 0.0
    for k in usable_keys:
        num = model_cost[k] - gold_cost[k]
        denom = baseline_cost[k] - gold_cost[k]
        # Per-definition normalization
        comp = 1.0 - (num / denom)
        score += (weights[k] / total_w) * comp

    return score


# ==========================================
# Convenience wrappers for dataset-level CF
# ==========================================
def cf_dataset_level(
    declared_list: List[Dict[str, str]],
    size_list: List[Sequence[float]],
    time_list: List[Sequence[float]],
    space_list: Optional[List[Sequence[float]]] = None
) -> float:
    """
    A thin alias for cf(), to emphasize dataset-level usage across N samples.
    """
    return cf(declared_list, size_list, time_list, space_list)
