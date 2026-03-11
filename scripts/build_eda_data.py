#!/usr/bin/env python3
"""
Build EDA outputs for notebooks/eda.qmd. Run outside Quarto (e.g. with venv activated).

  python scripts/build_eda_data.py [path/to/data.csv]

If no CSV path is given, uses the first data/raw/*.csv. Writes:
  notebooks/tables/  — hourly_metrics.csv, glycemic_metrics.csv, masking_summary.csv, timeseries_window.csv
  notebooks/images/  — masking_bar_chart.png, masking_timeseries.png

Then render the doc: quarto render notebooks/eda.qmd
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from matplotlib import ticker

from src import (
    compute_TIR,
    compute_TBR,
    compute_TAR,
    compute_GMI,
    compute_summary_metrics,
    local_variance_mask,
    jump_spike_mask,
    jitter_mask,
    drift_window_mask,
    dropout_flatline_mask,
    long_nan_run_mask,
    session_warmup_tail_mask,
    calibration_period_mask,
)
from src.hourly import make_hour_of_day_metrics, make_hour_dow_metrics


def main():
    # CSV path: arg or data/raw/*.csv
    if len(sys.argv) > 1:
        csv_path = Path(sys.argv[1])
    else:
        raw_dir = ROOT / "data" / "raw"
        files = list(raw_dir.glob("*.csv"))
        if not files:
            raise SystemExit("No CSV in data/raw and no path given")
        csv_path = files[0]

    tables_dir = ROOT / "notebooks" / "tables"
    images_dir = ROOT / "notebooks" / "images"
    tables_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    df_py = pd.read_csv(csv_path).dropna(axis=1, how="all")
    cgm_py = df_py[df_py["Event Type"] == "EGV"].copy().dropna(axis=1, how="all")
    calibrations_py = df_py[df_py["Event Type"] == "Calibration"].copy().dropna(axis=1, how="all")

    if cgm_py.shape[1] == 8:
        cgm_py.columns = [
            "index", "ts", "type", "subtype", "device", "egv", "device_tick", "device_id",
        ]
    if calibrations_py.shape[1] == 6:
        calibrations_py.columns = ["index", "ts", "type", "device", "egv", "device_id"]

    # 1. Hourly metrics
    hour_of_day = make_hour_of_day_metrics(cgm_py)
    hour_of_day.to_csv(tables_dir / "hourly_metrics.csv", index=False)

    # 2. Glycemic metrics (unmasked)
    glucose = cgm_py["egv"]
    raw = {
        "TIR": compute_TIR(glucose),
        "TBR": compute_TBR(glucose),
        "TAR": compute_TAR(glucose),
        "GMI": compute_GMI(glucose),
        "summary": compute_summary_metrics(glucose),
    }
    def round_summary(d):
        return {k: int(round(v, 0)) if np.isfinite(v) else v for k, v in d.items()}
    metrics_rows = []
    for k in ["TIR", "TBR", "TAR", "GMI"]:
        v = round(raw[k], 2)
        metrics_rows.append({"Metric": k, "Value": f"{100 * v:.1f}%" if k != "GMI" else f"{v:.2f}%"})
    for k, v in round_summary(raw["summary"]).items():
        label = k.capitalize() if k != "sd" else "SD"
        metrics_rows.append({"Metric": label, "Value": v if np.isfinite(v) else "—"})
    pd.DataFrame(metrics_rows).to_csv(tables_dir / "glycemic_metrics.csv", index=False)

    # 3. Masking summary
    glucose_arr = cgm_py["egv"]
    n = len(glucose_arr)
    m1 = local_variance_mask(glucose_arr, window_min=30, interval_min=5, threshold=None)
    m2 = jump_spike_mask(glucose_arr, threshold_mgdL=15, interval_min=5)
    m3 = jitter_mask(glucose_arr, window_min=30, min_sign_changes=4, interval_min=5)
    m4 = drift_window_mask(glucose_arr, drift_duration_hr=24, low_threshold_mgdL=70, low_duration_hr=8, interval_min=5)
    m5 = dropout_flatline_mask(glucose_arr, window_min=30, interval_min=5)
    m6 = long_nan_run_mask(glucose_arr, dropout_min=30, prior_hr=1, interval_min=5)
    m_warmup_tail = session_warmup_tail_mask(glucose_arr, cgm_py["device_id"], warmup_hr=24, tail_hr=24, interval_min=5)
    cal_ts = None
    for col in ("ts", "Timestamp", "Time"):
        if col in calibrations_py.columns:
            cal_ts = calibrations_py[col]
            break
    if cal_ts is None:
        raise SystemExit("Calibration table needs a timestamp column (ts, Timestamp, or Time)")
    m_calibration = calibration_period_mask(glucose_arr, cgm_py["ts"], cal_ts, prior_hr=1, post_min=30)

    def pad_mask(mask, size):
        if mask is None or size <= 0:
            return None
        m = np.asarray(mask, dtype=bool)
        return np.resize(m, size) if len(m) < size else m[:size]

    combined = pad_mask(m1, n) | pad_mask(m2, n) | pad_mask(m3, n) | pad_mask(m4, n) | pad_mask(m5, n) | pad_mask(m6, n) | pad_mask(m_warmup_tail, n) | pad_mask(m_calibration, n)
    scenarios = [
        ("Unmasked", None),
        ("Variance", pad_mask(m1, n)),
        ("Jump spike", pad_mask(m2, n)),
        ("Jitter", pad_mask(m3, n)),
        ("Drift", pad_mask(m4, n)),
        ("Flatline", pad_mask(m5, n)),
        ("Long NaN", pad_mask(m6, n)),
        ("Warmup/tail", pad_mask(m_warmup_tail, n)),
        ("Calibration", pad_mask(m_calibration, n)),
        ("Combined", pad_mask(combined, n)),
    ]
    rows = []
    for name, mask in scenarios:
        pct = 0.0 if mask is None else (100.0 * np.sum(mask) / n)
        tir = compute_TIR(glucose_arr, mask=mask)
        tbr = compute_TBR(glucose_arr, mask=mask)
        tar = compute_TAR(glucose_arr, mask=mask)
        gmi = compute_GMI(glucose_arr, mask=mask)
        sm = compute_summary_metrics(glucose_arr, mask=mask)
        rows.append({
            "Scenario": name,
            "pct_flagged": round(pct, 2),
            "TIR": round(tir, 2) if np.isfinite(tir) else np.nan,
            "TBR": round(tbr, 2) if np.isfinite(tbr) else np.nan,
            "TAR": round(tar, 2) if np.isfinite(tar) else np.nan,
            "GMI": round(gmi, 2) if np.isfinite(gmi) else np.nan,
            "Mean_mg_dL": int(round(sm["mean"], 0)) if np.isfinite(sm["mean"]) else np.nan,
        })
    masking_df = pd.DataFrame(rows)
    masking_df.to_csv(tables_dir / "masking_summary.csv", index=False)

    # 4. Bar chart
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    for ax, col in zip(axes, ["TIR", "TBR", "TAR", "GMI"]):
        bars = ax.bar(range(len(masking_df)), masking_df[col], width=0.6, color="steelblue", edgecolor="white", linewidth=0.5)
        ax.set_xticks(range(len(masking_df)))
        ax.set_xticklabels(masking_df["Scenario"], rotation=45, ha="right")
        ax.set_ylabel(col)
        ax.set_title(col)
        if col == "GMI":
            ax.set_ylim(4, 7)
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f"{x:.1f}%"))
            ax.bar_label(bars, labels=[f"{v:.1f}%" if np.isfinite(v) else "—" for v in masking_df[col]])
        else:
            ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
            ax.bar_label(bars, fmt="{:.0%}")
    plt.suptitle("Metric sensitivity by masking scenario", y=1.02)
    plt.tight_layout()
    plt.savefig(images_dir / "masking_bar_chart.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 5. Time-series window
    window_points = min(5 * 24 * 12, len(glucose_arr))
    s_win = cgm_py["egv"].iloc[:window_points]
    glucose_win = np.asarray(s_win.replace("Low", 39) if hasattr(s_win, "replace") else [39 if x == "Low" else x for x in s_win], dtype=float)
    mask_win = combined[:window_points]
    ts_window = pd.DataFrame({"index": np.arange(window_points), "glucose": glucose_win, "unstable": mask_win})
    ts_window.to_csv(tables_dir / "timeseries_window.csv", index=False)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(ts_window["index"], ts_window["glucose"], color="black", linewidth=0.6, alpha=0.9, label="Glucose")
    unstarts = np.where(np.diff(np.concatenate([[False], mask_win, [False]]).astype(int)) == 1)[0]
    unends = np.where(np.diff(np.concatenate([[False], mask_win, [False]]).astype(int)) == -1)[0]
    for i, (start, end) in enumerate(zip(unstarts, unends)):
        ax.axvspan(start, end, alpha=0.25, color="red", label="Unstable" if i == 0 else None)
    ax.set_xlabel("Time index (5-min readings)")
    ax.set_ylabel("Glucose (mg/dL)")
    ax.set_title("Glucose and combined instability mask (first 5 days)")
    ax.set_ylim(40, 350)
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(images_dir / "masking_timeseries.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("Wrote", tables_dir, "and", images_dir)


if __name__ == "__main__":
    main()
