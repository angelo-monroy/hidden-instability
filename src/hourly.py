import pandas as pd

from .metrics import (
    compute_TIR,
    compute_TBR,
    compute_TAR,
    compute_GMI,
    compute_summary_metrics,
)


def make_hourly_metrics(cgm_py: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate CGM data to hourly metrics.

    Expects a DataFrame with at least:
      - 'ts': timestamp column
      - 'egv': glucose values (mg/dL)

    Returns a DataFrame with one row per hour and:
      - ts (hour start)
      - n_points, mean, sd, cv, median, min, max
      - TIR, TBR, TAR, GMI
    """
    cgm_py = cgm_py.copy()
    cgm_py["ts"] = pd.to_datetime(cgm_py["ts"])
    cgm_py = cgm_py.sort_values("ts")

    def _hourly_metrics(group: pd.DataFrame) -> pd.Series:
        # Pass the raw Series through; metrics functions normalize "Low" -> 39.
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

    hourly = (
        cgm_py.groupby(pd.Grouper(key="ts", freq="1h"))
        .apply(_hourly_metrics)
        .reset_index()
    )
    return hourly

