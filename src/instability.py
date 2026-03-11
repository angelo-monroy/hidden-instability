"""
Instability heuristics for CGM time series (5-min readings).
All functions take a 1D array of glucose values and return a boolean mask
(True = unstable). Combine with instability_mask() for the full derived mask.
Accepts raw EGV (e.g. Dexcom "Low" strings) and normalizes like metrics.
"""

import numpy as np
import pandas as pd

from .metrics import _normalize_glucose


def _as_array(series):
    """Extract 1D float array from Series or array; normalize 'Low' like metrics."""
    series = _normalize_glucose(series)
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


def jitter_mask(glucose, window_min=30, min_sign_changes=2, interval_min=5):
    """
    Flag periods with too much small oscillation (jitter)—many direction reversals
    even when individual steps are small (e.g. 5–10 mg/dL up and down).

    Good CGM data looks like a sequential line; jittery data oscillates around an
    imaginary average with no smooth trend. This mask counts sign changes in
    first differences over a window: high reversal count = scatter / no pattern,
    low = dotted line. Entire window is marked unstable when sign changes
    >= min_sign_changes (default 2 in a 30-min window).
    """
    arr = _as_array(glucose)
    n = arr.size
    k = max(3, int(window_min / interval_min))  # need at least 3 points for 2 diffs
    if n < k:
        return np.zeros(n, dtype=bool)

    out = np.zeros(n, dtype=bool)
    for i in range(k - 1, n):
        w = arr[i - k + 1 : i + 1]
        if not np.all(np.isfinite(w)):
            continue
        d = np.diff(w)
        if d.size < 2:
            continue
        # count direction reversals: sign change between consecutive diffs
        sign_changes = np.sum((d[1:] * d[:-1]) < 0)
        if sign_changes >= min_sign_changes:
            out[i - k + 1 : i + 1] = True
    return out


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
       Flagged as potential sensor drift (reading low), overnight compression low, or prolonged hypo.

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
    Flag two patterns:

    1. Flatline: repeated identical readings over at least window_min (e.g. 6 same
       values @ 5 min). Same value held for 30+ min can indicate sensor stuck.

    2. Dropout: any reading that is NaN. When a sensor fails or drops out we get
       NaNs; the occasional NaN between two numeric values is still dropout and
       is flagged. Every NaN is marked unstable.

    Session-level context: max session length depends on device—use
    max_session_days(device_id) from src.session (G7 → 10.5 days, G6 → 10 days).
    """
    arr = _as_array(glucose)
    n = arr.size

    # 1. Flatline: same value for window_min or longer
    out = np.zeros(n, dtype=bool)
    k = max(2, int(window_min / interval_min))
    if n >= k:
        for i in range(k - 1, n):
            w = arr[i - k + 1 : i + 1]
            if np.all(np.isfinite(w)) and np.nanmax(w) == np.nanmin(w):
                out[i - k + 1 : i + 1] = True

    # 2. Dropout: any NaN (sensor stopped or occasional missing value)
    out = out | (~np.isfinite(arr))
    return out


def long_nan_run_mask(glucose, dropout_min=30, prior_hr=1, interval_min=5):
    """
    Separate mask: long NaN runs (>= dropout_min) and the prior_hr before each.

    When a NaN run is at least dropout_min (default 30 min), flag the entire run
    and the prior_hr (default 1 hour) before the run starts—the lead-up can be
    unreliable. Short NaN runs are not flagged by this mask (use dropout_flatline
    for any NaN). Combine with other masks via OR in instability_mask.
    """
    arr = _as_array(glucose)
    n = arr.size
    is_nan = ~np.isfinite(arr)
    k_dropout = max(1, int(dropout_min / interval_min))
    k_prior = max(0, int(prior_hr * 60 / interval_min))

    out = np.zeros(n, dtype=bool)
    run_start = None
    for i in range(n + 1):
        if i < n and is_nan[i]:
            if run_start is None:
                run_start = i
        else:
            if run_start is not None:
                run_end = i
                run_len = run_end - run_start
                if run_len >= k_dropout:
                    out[run_start:run_end] = True
                    prior_start = max(0, run_start - k_prior)
                    out[prior_start:run_start] = True
                run_start = None
    return out


def session_warmup_tail_mask(
    glucose,
    device_id,
    *,
    warmup_hr=24,
    tail_hr=24,
    interval_min=5,
):
    """
    Flag the first warmup_hr and last tail_hr of each CGM session (contiguous device_id).

    Dexcom exports omit rows when the sensor is off, so there are no NaNs for "CGM off".
    Sessions are inferred from device_id: a new device_id starts a new session. The first
    ~24 h of a new sensor are often less accurate (warm-up); the last ~24 h before swap are
    also less reliable. This mask flags those segments so they can be excluded from metrics.

    Parameters
    ----------
    glucose : array-like
        1D glucose series (length used for alignment; values not used).
    device_id : array-like
        Same length as glucose; identifies the sensor/session per reading (e.g. cgm_py["device_id"]).
    warmup_hr : float
        Hours at the start of each session to flag (default 24).
    tail_hr : float
        Hours at the end of each session to flag (default 24).
    interval_min : int
        Minutes between readings (default 5); used to convert hours to point counts.

    Returns
    -------
    np.ndarray
        Boolean mask, True = unstable (warm-up or tail of a session).
    """
    n = len(np.asarray(glucose).reshape(-1))
    device_id = np.asarray(device_id).reshape(-1)
    if device_id.size != n:
        raise ValueError("device_id length must match glucose length")

    points_per_hr = max(1, int(60 / interval_min))
    warmup_points = int(warmup_hr * points_per_hr)
    tail_points = int(tail_hr * points_per_hr)

    # Run boundaries: new run where device_id changes (or at index 0)
    run_start = np.zeros(n, dtype=bool)
    run_start[0] = True
    run_start[1:] = device_id[1:] != device_id[:-1]
    run_start_ix = np.where(run_start)[0]
    run_end_ix = np.concatenate([run_start_ix[1:] - 1, [n - 1]])

    out = np.zeros(n, dtype=bool)
    for start, end in zip(run_start_ix, run_end_ix):
        # First warmup_hr of session
        first_end = min(start + warmup_points, end + 1)
        out[start:first_end] = True
        # Last tail_hr of session
        last_start = max(end - tail_points + 1, start)
        out[last_start : end + 1] = True
    return out


def calibration_period_mask(
    glucose,
    cgm_ts,
    calibration_ts,
    *,
    prior_hr=1,
    post_min=30,
):
    """
    Flag CGM readings in a window around each manual BGM calibration event.

    Calibrations indicate user-specified instability: the period leading up to
    (fingerstick / meter entry) and following (sensor adjusting) the event is
    often less reliable. This mask flags [t_cal - prior_hr, t_cal + post_min]
    for each calibration time so those segments can be excluded from metrics.

    Parameters
    ----------
    glucose : array-like
        1D glucose series (length used for alignment).
    cgm_ts : array-like
        Timestamps for each CGM reading, same length as glucose (e.g. cgm_py["ts"]).
    calibration_ts : array-like
        Timestamps of calibration events (e.g. calibrations_py["ts"] or the ts column).
    prior_hr : float
        Hours before each calibration to flag (default 1).
    post_min : float
        Minutes after each calibration to flag (default 30).

    Returns
    -------
    np.ndarray
        Boolean mask, True = unstable (within any calibration window).
    """
    n = len(np.asarray(glucose).reshape(-1))
    cgm_ts = pd.to_datetime(cgm_ts).values
    calibration_ts = pd.to_datetime(calibration_ts).values
    if cgm_ts.size != n:
        raise ValueError("cgm_ts length must match glucose length")

    out = np.zeros(n, dtype=bool)
    for t_cal in calibration_ts:
        t_cal = pd.Timestamp(t_cal)
        start = t_cal - pd.Timedelta(hours=prior_hr)
        end = t_cal + pd.Timedelta(minutes=post_min)
        out |= (cgm_ts >= start) & (cgm_ts <= end)
    return out


def instability_mask(
    glucose,
    *,
    variance_threshold=None,
    jump_threshold_mgdL=20,
    jitter_window_min=30,
    min_sign_changes=2,
    drift_duration_hr=24,
    low_threshold_mgdL=70,
    low_duration_hr=8,
    flatline_window_min=30,
    dropout_min=30,
    prior_hr=1,
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
    m3 = jitter_mask(
        arr,
        window_min=jitter_window_min,
        min_sign_changes=min_sign_changes,
        interval_min=interval_min,
    )
    m4 = drift_window_mask(
        arr,
        drift_duration_hr=drift_duration_hr,
        low_threshold_mgdL=low_threshold_mgdL,
        low_duration_hr=low_duration_hr,
        interval_min=interval_min,
    )
    m5 = dropout_flatline_mask(arr, window_min=flatline_window_min, interval_min=interval_min)
    m6 = long_nan_run_mask(arr, dropout_min=dropout_min, prior_hr=prior_hr, interval_min=interval_min)

    # ensure same length (edge effects can differ by 1)
    def pad(b, size):
        if len(b) < size:
            return np.resize(b, size)
        return b[:size]

    m1, m2, m3, m4, m5, m6 = (pad(m, n) for m in (m1, m2, m3, m4, m5, m6))
    return (m1 | m2 | m3 | m4 | m5 | m6).astype(bool)
