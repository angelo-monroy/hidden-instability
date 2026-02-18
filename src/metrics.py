"""
CGM metrics: TIR, TBR, TAR. Optional mask to exclude unstable segments.
Conventions: TIR 70–180 mg/dL, TBR <70, TAR >180.
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
