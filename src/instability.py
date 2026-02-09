"""
Instability heuristics for CGM time series (5-min readings).
All functions take a 1D array of glucose values and return a boolean mask
(True = unstable). Combine with instability_mask() for the full derived mask.
"""

import numpy as np


def _as_array(series):
    """Extract 1D float array from Series or array; drop NaN for window logic."""
    arr = np.asarray(series, dtype=float)
    if arr.ndim != 1:
        raise ValueError("Expected 1D glucose series")
    return arr


def local_variance_mask(glucose, window_min=30, interval_min=5, threshold=None):
    """
    Rolling variance over window_min; flag windows where variance > threshold.
    Default threshold: 95th percentile of rolling variance (adaptive).
    """
    arr = _as_array(glucose)
    n = arr.size
    k = max(1, int(window_min / interval_min))  # e.g. 6 for 30 min @ 5 min
    if n < k:
        return np.zeros(n, dtype=bool)

    # rolling variance (skip NaN in window for robustness)
    var = np.full(n, np.nan)
    for i in range(k - 1, n):
        w = arr[i - k + 1 : i + 1]
        if np.any(np.isfinite(w)):
            var[i] = np.nanvar(w)
    if threshold is None:
        threshold = np.nanpercentile(var[~np.isnan(var)], 95)
    return np.where(np.isfinite(var), var > threshold, False)


def jump_spike_mask(glucose, threshold_mgdL=20, interval_min=5):
    """
    Consecutive readings with absolute change > threshold_mgdL within one interval.
    """
    arr = _as_array(glucose)
    diff = np.abs(np.diff(arr, prepend=arr[0]))
    return diff > threshold_mgdL


def drift_window_mask(glucose, window_hr=3, interval_min=5, threshold_mgdL=30):
    """
    Monotonic deviation over window_hr exceeding threshold_mgdL
    (max - min in window).
    """
    arr = _as_array(glucose)
    n = arr.size
    k = max(2, int(window_hr * 60 / interval_min))
    if n < k:
        return np.zeros(n, dtype=bool)

    out = np.zeros(n, dtype=bool)
    for i in range(k - 1, n):
        w = arr[i - k + 1 : i + 1]
        if np.any(np.isfinite(w)):
            r = np.nanmax(w) - np.nanmin(w)
            if r >= threshold_mgdL:
                out[i - k + 1 : i + 1] = True
    return out


def dropout_flatline_mask(glucose, window_min=30, interval_min=5):
    """
    Repeated identical readings over at least window_min (e.g. 6 same values @ 5 min).
    """
    arr = _as_array(glucose)
    n = arr.size
    k = max(2, int(window_min / interval_min))
    if n < k:
        return np.zeros(n, dtype=bool)

    out = np.zeros(n, dtype=bool)
    for i in range(k - 1, n):
        w = arr[i - k + 1 : i + 1]
        if np.all(np.isfinite(w)) and np.nanmax(w) == np.nanmin(w):
            out[i - k + 1 : i + 1] = True
    return out


def instability_mask(
    glucose,
    *,
    variance_threshold=None,
    jump_threshold_mgdL=20,
    drift_window_hr=3,
    drift_threshold_mgdL=30,
    flatline_window_min=30,
    interval_min=5,
):
    """
    Combined binary instability mask: True where any heuristic flags.
    Aligned with input length. Use this as the derived instability mask
    for metric sensitivity (TIR/TBR/TAR masked vs unmasked).
    """
    arr = _as_array(glucose)
    n = len(arr)

    m1 = local_variance_mask(arr, window_min=30, interval_min=interval_min, threshold=variance_threshold)
    m2 = jump_spike_mask(arr, threshold_mgdL=jump_threshold_mgdL, interval_min=interval_min)
    m3 = drift_window_mask(arr, window_hr=drift_window_hr, interval_min=interval_min, threshold_mgdL=drift_threshold_mgdL)
    m4 = dropout_flatline_mask(arr, window_min=flatline_window_min, interval_min=interval_min)

    # ensure same length (edge effects can differ by 1)
    def pad(b, size):
        if len(b) < size:
            return np.resize(b, size)
        return b[:size]

    m1, m2, m3, m4 = (pad(m, n) for m in (m1, m2, m3, m4))
    return (m1 | m2 | m3 | m4).astype(bool)
