"""Shared harness statistics: 95% CI (Student t) and Tukey IQR outlier indices.

Mirrors `crates/bench/src/lib.rs` so PyTorch baselines match Rust/Candle JSON fields.
"""
from __future__ import annotations

import math
import statistics
from typing import List, MutableMapping, Optional, Tuple

# Two-sided 97.5% quantile of Student's t, df = 1..30; else normal 1.96.
_T975_TABLE: Tuple[float, ...] = (
    12.706204,
    4.302653,
    3.182446,
    2.776445,
    2.570582,
    2.446912,
    2.364624,
    2.306004,
    2.262157,
    2.228139,
    2.200985,
    2.178813,
    2.160369,
    2.144787,
    2.131450,
    2.119905,
    2.109816,
    2.100922,
    2.093024,
    2.085963,
    2.079614,
    2.073873,
    2.068658,
    2.063899,
    2.059539,
    2.055529,
    2.051831,
    2.048407,
    2.045230,
    2.042272,
)


def t_crit_975(df: int) -> float:
    if df <= 0:
        return float("nan")
    if df > 30:
        return 1.96
    return _T975_TABLE[df - 1]


def linear_percentile(sorted_vals: List[float], pct: float) -> float:
    n = len(sorted_vals)
    if n == 0:
        return 0.0
    if n == 1:
        return sorted_vals[0]
    pos = (n - 1) * (pct / 100.0)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_vals[lo]
    w = pos - lo
    return sorted_vals[lo] * (1.0 - w) + sorted_vals[hi] * w


def ci95_mean_ms(runs_ms: List[float]) -> Optional[Tuple[float, float]]:
    n = len(runs_ms)
    if n < 2:
        return None
    mean = statistics.mean(runs_ms)
    s = statistics.stdev(runs_ms)
    if not math.isfinite(s):
        return None
    t = t_crit_975(n - 1)
    half = t * s / math.sqrt(n)
    return (max(0.0, mean - half), mean + half)


def tukey_iqr_outlier_indices(runs_ms: List[float]) -> List[int]:
    if not runs_ms:
        return []
    sorted_r = sorted(runs_ms)
    q1 = linear_percentile(sorted_r, 25.0)
    q3 = linear_percentile(sorted_r, 75.0)
    iqr = q3 - q1
    low = q1 - 1.5 * iqr
    high = q3 + 1.5 * iqr
    return [i for i, v in enumerate(runs_ms) if v < low or v > high]


def enrich_sample_stats(sample: MutableMapping[str, object]) -> None:
    """Add optional ci95_*, outlier_run_indices, stats_note to a schema-v2 sample dict."""
    runs = sample.get("runs_ms")
    if not isinstance(runs, list) or not runs:
        return
    runs_f = [float(x) for x in runs]
    ci = ci95_mean_ms(runs_f)
    if ci is not None:
        sample["ci95_low_ms"] = ci[0]
        sample["ci95_high_ms"] = ci[1]
        sample["stats_note"] = "t_interval_on_ms;iqr_outliers"
    outs = tukey_iqr_outlier_indices(runs_f)
    if outs:
        sample["outlier_run_indices"] = outs
