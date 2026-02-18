"""
CGM instability and metrics pipeline.
Derived layer: heuristics + mask. Metrics: TIR, TBR, TAR with optional masking.
"""

from .instability import (
    local_variance_mask,
    jump_spike_mask,
    jitter_mask,
    drift_window_mask,
    dropout_flatline_mask,
    long_nan_run_mask,
    instability_mask,
)
from .metrics import compute_TIR, compute_TBR, compute_TAR, compute_summary_metrics
from .session import max_session_days

__all__ = [
    "local_variance_mask",
    "jump_spike_mask",
    "jitter_mask",
    "drift_window_mask",
    "dropout_flatline_mask",
    "long_nan_run_mask",
    "instability_mask",
    "compute_TIR",
    "compute_TBR",
    "compute_TAR",
    "compute_summary_metrics",
    "max_session_days",
]
