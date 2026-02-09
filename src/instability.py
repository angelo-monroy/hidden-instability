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


def drift_window_mask(
    glucose,
    *,
    drift_duration_hr=24,
    low_threshold_mgdL=70,
    low_duration_hr=8,
    interval_min=5,
):
    """
    Flag two patterns that may indicate sensor drift or governance blind spots
    (not physiologic swings from insulin/carbs):

    1. Monotonic drift > drift_duration_hr (default 24 h): glucose drifts in one
       direction for longer than a day. Possible sensor drift or untreated
       elevation; flagged as potential instability, not verified.

    2. Below range for > low_duration_hr (default 8 h): glucose is below
       low_threshold_mgdL (default 70) and stays there for more than 8 hours.
       Flagged as potential sensor drift (reading low) or prolonged hypo.

    Cost: monotonic check is O(n * window_length) with window = 24h of points;
    low run is O(n). For 30 days @ 5 min, ~2.5e6 ops for monotonic, negligible for low run.
    """
    arr = _as_array(glucose)
    n = arr.size
    points_per_hr = int(60 / interval_min)
    k_drift = max(2, int(drift_duration_hr * points_per_hr))   # e.g. 288 for 24 h @ 5 min
    k_low = max(2, int(low_duration_hr * points_per_hr))       # e.g. 96 for 8 h @ 5 min

    out = np.zeros(n, dtype=bool)

    # 1. Monotonic drift longer than drift_duration_hr
    if n >= k_drift:
        for i in range(k_drift - 1, n):
            w = arr[i - k_drift + 1 : i + 1]
            if not np.all(np.isfinite(w)):
                continue
            d = np.diff(w)
            if (d >= 0).all() or (d <= 0).all():
                out[i - k_drift + 1 : i + 1] = True

    # 2. Below low_threshold_mgdL for longer than low_duration_hr
    below = (arr < low_threshold_mgdL) & np.isfinite(arr)
    if np.any(below):
        run_start = None
        for i in range(n + 1):
            if i < n and below[i]:
                if run_start is None:
                    run_start = i
            else:
                if run_start is not None:
                    run_len = i - run_start
                    if run_len > k_low:
                        out[run_start:i] = True
                    run_start = None

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
    drift_duration_hr=24,
    low_threshold_mgdL=70,
    low_duration_hr=8,
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
    m3 = drift_window_mask(
        arr,
        drift_duration_hr=drift_duration_hr,
        low_threshold_mgdL=low_threshold_mgdL,
        low_duration_hr=low_duration_hr,
        interval_min=interval_min,
    )
    m4 = dropout_flatline_mask(arr, window_min=flatline_window_min, interval_min=interval_min)

    # ensure same length (edge effects can differ by 1)
    def pad(b, size):
        if len(b) < size:
            return np.resize(b, size)
        return b[:size]

    m1, m2, m3, m4 = (pad(m, n) for m in (m1, m2, m3, m4))
    return (m1 | m2 | m3 | m4).astype(bool)
