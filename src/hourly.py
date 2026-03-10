import pandas as pd

from .metrics import (
    compute_TIR,
    compute_TBR,
    compute_TAR,
    compute_GMI,
    compute_summary_metrics,
)


def _metrics_for_group(group: pd.DataFrame) -> pd.Series:
    """
    Common per-group metric computation on a subset of CGM rows.
    Expects an 'egv' column that may contain the string 'Low'.
    """
    glucose = group["egv"]
    summary = compute_summary_metrics(glucose)
    return pd.Series(
        {
            "n_points": int(glucose.shape[0]),
            "mean": summary["mean"],
            "sd": summary["sd"],
            "cv": summary["cv"],
            "median": summary["median"],
            "min": summary["min"],
            "max": summary["max"],
            "TIR": compute_TIR(glucose),
            "TBR": compute_TBR(glucose),
            "TAR": compute_TAR(glucose),
            "GMI": compute_GMI(glucose),
        }
    )


def make_hourly_metrics(cgm_py: pd.DataFrame) -> pd.DataFrame:
    """
    Time-series hourly aggregation (one row per clock hour in time).
    Kept for completeness, but for diurnal patterns prefer the
    by-hour-of-day helpers below.
    """
    cgm_py = cgm_py.copy()
    ts = pd.to_datetime(cgm_py["ts"])
    # Localize naive timestamps to Pacific time (PST/PDT as appropriate).
    # Handle DST fall-back (ambiguous times) and spring-forward (nonexistent times)
    # deterministically so aggregation does not fail.
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize(
            "America/Los_Angeles",
            ambiguous="NaT",        # mark repeated 1–2 AM times as NaT
            nonexistent="shift_forward",  # shift spring-forward gaps forward
        )
    else:
        ts = ts.dt.tz_convert("America/Los_Angeles")
    cgm_py["ts"] = ts
    cgm_py = cgm_py.sort_values("ts")

    hourly = (
        cgm_py.groupby(pd.Grouper(key="ts", freq="1h"))
        .apply(_metrics_for_group)
        .reset_index()
    )
    return hourly


def make_hour_of_day_metrics(cgm_py: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse across all days to get a diurnal profile:
    one row per hour-of-day (0–23) with CGM metrics.
    """
    cgm_py = cgm_py.copy()
    ts = pd.to_datetime(cgm_py["ts"])
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize(
            "America/Los_Angeles",
            ambiguous="NaT",
            nonexistent="shift_forward",
        )
    else:
        ts = ts.dt.tz_convert("America/Los_Angeles")
    cgm_py["ts"] = ts
    # Integer hour-of-day (0–23) plus a time-of-day label for easy coercion in R
    cgm_py["hour"] = cgm_py["ts"].dt.hour
    cgm_py["hour_time"] = cgm_py["ts"].dt.strftime("%H:%M:%S")

    by_hour = (
        cgm_py.groupby("hour")
        .apply(_metrics_for_group)
        .reset_index()
        .sort_values("hour")
    )
    return by_hour


def make_hour_dow_metrics(cgm_py: pd.DataFrame) -> pd.DataFrame:
    """
    Diurnal profile stratified by day-of-week:
    one row per (day_of_week, hour) combination.
    """
    cgm_py = cgm_py.copy()
    ts = pd.to_datetime(cgm_py["ts"])
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize(
            "America/Los_Angeles",
            ambiguous="NaT",
            nonexistent="shift_forward",
        )
    else:
        ts = ts.dt.tz_convert("America/Los_Angeles")
    cgm_py["ts"] = ts
    # Integer hour-of-day plus time-of-day label
    cgm_py["hour"] = cgm_py["ts"].dt.hour
    cgm_py["hour_time"] = cgm_py["ts"].dt.strftime("%H:%M:%S")
    cgm_py["day_of_week"] = cgm_py["ts"].dt.day_name()

    by_hour_dow = (
        cgm_py.groupby(["day_of_week", "hour"])
        .apply(_metrics_for_group)
        .reset_index()
        .sort_values(["day_of_week", "hour"])
    )
    return by_hour_dow
