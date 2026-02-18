"""
CGM metrics: TIR, TBR, TAR, GMI, summary (mean, SD, CV, median, min, max).
Optional mask to exclude unstable segments.
Conventions: TIR 70–180 mg/dL, TBR <70, TAR >180.
GMI: estimated A1C-equivalent (%) from mean glucose (mg/dL).
"""

import numpy as np


def _as_array(series):
    arr = np.asarray(series, dtype=float)
    if arr.ndim != 1:
        raise ValueError("Expected 1D glucose series")
    return arr


def _masked_series(glucose, mask):
    """If mask is not None, exclude masked indices from numerator and denominator."""
    arr = _as_array(glucose)
    if mask is None:
        return arr, np.ones(arr.size, dtype=bool)
    m = np.asarray(mask, dtype=bool)
    if m.size != arr.size:
        raise ValueError("mask length must match glucose length")
    use = ~m
    return arr, use


def compute_TIR(glucose, mask=None, low=70, high=180):
    """
    Time in range [low, high] as fraction of total (used) time.
    If mask is provided, excluded indices are omitted from both numerator and denominator.
    """
    arr, use = _masked_series(glucose, mask)
    if not np.any(use):
        return np.nan
    valid = np.isfinite(arr) & use
    if not np.any(valid):
        return np.nan
    in_range = (arr >= low) & (arr <= high) & use
    return np.sum(in_range) / np.sum(use)


def compute_TBR(glucose, mask=None, low=70):
    """
    Time below range (< low) as fraction of total (used) time.
    """
    arr, use = _masked_series(glucose, mask)
    if not np.any(use):
        return np.nan
    below = (arr < low) & use
    return np.sum(below) / np.sum(use)


def compute_TAR(glucose, mask=None, high=180):
    """
    Time above range (> high) as fraction of total (used) time.
    """
    arr, use = _masked_series(glucose, mask)
    if not np.any(use):
        return np.nan
    above = (arr > high) & use
    return np.sum(above) / np.sum(use)


def compute_GMI(glucose, mask=None):
    """
    Glucose Management Indicator: estimated A1C-equivalent (%) from mean glucose (mg/dL).
    Formula: GMI = 3.31 + 0.02392 × mean_glucose (Bergenstal et al.).
    If mask is provided, excluded indices are omitted; mean is over used, finite values only.
    """
    arr, use = _masked_series(glucose, mask)
    valid = np.isfinite(arr) & use
    if not np.any(valid):
        return np.nan
    mean_glucose = float(np.mean(arr[valid]))
    return 3.31 + 0.02392 * mean_glucose


def compute_summary_metrics(glucose, mask=None):
    """
    Summary statistics over used, finite glucose values: mean, SD, CV, median, min, max.
    If mask is provided, excluded indices are omitted. CV = SD/mean (ratio); NaN if mean is 0.
    Returns a dict with keys: mean, sd, cv, median, min, max.
    """
    arr, use = _masked_series(glucose, mask)
    valid = np.isfinite(arr) & use
    if not np.any(valid):
        return {
            "mean": np.nan,
            "sd": np.nan,
            "cv": np.nan,
            "median": np.nan,
            "min": np.nan,
            "max": np.nan,
        }
    x = arr[valid]
    mean = float(np.mean(x))
    sd = float(np.std(x, ddof=1))
    cv = sd / mean if mean != 0 else np.nan
    return {
        "mean": mean,
        "sd": sd,
        "cv": cv,
        "median": float(np.median(x)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
    }
