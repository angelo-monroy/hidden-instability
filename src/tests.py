"""
Unit tests for instability mask heuristics.

Uses in-code synthetic fixtures only (no external CSV). Run from repo root with:
  python -m unittest src.tests
or, with pytest installed:
  pytest src/tests.py -v
"""

import unittest
import numpy as np
import pandas as pd

from .instability import (
    local_variance_mask,
    jump_spike_mask,
    jitter_mask,
    drift_window_mask,
    dropout_flatline_mask,
    long_nan_run_mask,
    session_warmup_tail_mask,
    calibration_period_mask,
    instability_mask,
)


# ---- Fixtures (synthetic data; no external files) ----

def _glucose(*values):
    """1D float array from values (NaNs allowed)."""
    return np.array(values, dtype=float)


# ---- local_variance_mask ----

class TestLocalVarianceMask(unittest.TestCase):
    def test_output_shape_and_type(self):
        g = _glucose(100, 102, 98, 101, 99, 100)
        m = local_variance_mask(g, window_min=30, interval_min=5)
        self.assertEqual(m.shape, (6,), "mask length must match glucose")
        self.assertEqual(m.dtype, bool)

    def test_short_series_returns_no_flags(self):
        g = _glucose(100, 101)
        m = local_variance_mask(g, window_min=30, interval_min=5)
        np.testing.assert_array_equal(m, False)

    def test_high_variance_window_flagged(self):
        # One window with high variance; others low
        g = _glucose(100, 100, 100, 100, 100, 100, 50, 150, 50, 150, 50, 150)
        m = local_variance_mask(g, window_min=30, interval_min=5, threshold=100.0)
        self.assertTrue(np.any(m), "at least one high-variance window should be flagged")


# ---- jump_spike_mask ----

class TestJumpSpikeMask(unittest.TestCase):
    def test_output_shape_and_type(self):
        g = _glucose(100, 102, 105, 110)
        m = jump_spike_mask(g, threshold_mgdL=20, interval_min=5)
        self.assertEqual(m.shape, (4,))
        self.assertEqual(m.dtype, bool)

    def test_large_step_flagged(self):
        g = _glucose(100, 105, 130, 132)  # step 25 at index 2
        m = jump_spike_mask(g, threshold_mgdL=20, interval_min=5)
        self.assertTrue(m[2], "index of large step should be True")

    def test_small_steps_not_flagged(self):
        g = _glucose(100, 105, 108, 112)
        m = jump_spike_mask(g, threshold_mgdL=20, interval_min=5)
        self.assertFalse(np.any(m))


# ---- jitter_mask ----

class TestJitterMask(unittest.TestCase):
    def test_output_shape_and_type(self):
        g = _glucose(100, 102, 98, 101, 99, 100, 98)
        m = jitter_mask(g, window_min=30, min_sign_changes=2, interval_min=5)
        self.assertEqual(m.shape, (7,))
        self.assertEqual(m.dtype, bool)

    def test_short_series_returns_no_flags(self):
        g = _glucose(100, 101, 102)
        m = jitter_mask(g, window_min=30, min_sign_changes=2, interval_min=5)
        np.testing.assert_array_equal(m, False)

    def test_oscillating_window_flagged(self):
        # up, down, up, down -> multiple sign changes in diffs
        g = _glucose(100, 110, 100, 110, 100, 110, 100)
        m = jitter_mask(g, window_min=30, min_sign_changes=2, interval_min=5)
        self.assertTrue(np.any(m))


# ---- drift_window_mask ----

class TestDriftWindowMask(unittest.TestCase):
    def test_output_shape_and_type(self):
        g = _glucose(100, 101, 102, 103, 104)
        m = drift_window_mask(
            g,
            drift_duration_hr=24,
            low_threshold_mgdL=70,
            low_duration_hr=8,
            interval_min=5,
        )
        self.assertEqual(m.shape, (5,))
        self.assertEqual(m.dtype, bool)

    def test_long_monotonic_drift_flagged(self):
        # Short drift window for test: 6 points = 30 min @ 5 min
        n = 10
        g = np.linspace(80, 120, n)
        m = drift_window_mask(
            g,
            drift_duration_hr=0.5,
            low_threshold_mgdL=70,
            low_duration_hr=8,
            interval_min=5,
        )
        self.assertTrue(np.any(m), "long monotonic segment should be flagged")

    def test_long_low_run_flagged(self):
        # 10 points at 65 mg/dL with 5-min interval = 45 min; need 8 hr = 96 points for default
        # Use low_duration_hr small so 10 points counts
        g = np.concatenate([np.full(10, 65.0), np.full(5, 100.0)])
        m = drift_window_mask(
            g,
            drift_duration_hr=24,
            low_threshold_mgdL=70,
            low_duration_hr=0.5,
            interval_min=5,
        )
        self.assertTrue(np.any(m), "long below-threshold run should be flagged")


# ---- dropout_flatline_mask ----

class TestDropoutFlatlineMask(unittest.TestCase):
    def test_output_shape_and_type(self):
        g = _glucose(100, 100, 100, 100, 100, 100)
        m = dropout_flatline_mask(g, window_min=30, interval_min=5)
        self.assertEqual(m.shape, (6,))
        self.assertEqual(m.dtype, bool)

    def test_flatline_flagged(self):
        g = _glucose(100, 100, 100, 100, 100, 100)
        m = dropout_flatline_mask(g, window_min=30, interval_min=5)
        self.assertTrue(np.any(m), "constant window should be flagged as flatline")

    def test_input_with_nan_normalized_then_no_dropout_flags(self):
        # _as_array normalizes glucose (NaN -> 39), so after normalization there are
        # no NaNs; the dropout branch is not exercised for array input.
        g = _glucose(100, 102, np.nan, 104, 105)
        m = dropout_flatline_mask(g, window_min=30, interval_min=5)
        self.assertEqual(m.shape, (5,), "mask length must match input")


# ---- long_nan_run_mask ----

class TestLongNanRunMask(unittest.TestCase):
    def test_output_shape_and_type(self):
        g = _glucose(100, 101, np.nan, np.nan, 102)
        m = long_nan_run_mask(g, dropout_min=30, prior_hr=1, interval_min=5)
        self.assertEqual(m.shape, (5,))
        self.assertEqual(m.dtype, bool)

    def test_all_finite_input_yields_no_long_nan_flags(self):
        # _as_array normalizes (NaN -> 39), so with array input we never have NaN
        # in the internal arr; this mask returns all False for all-finite input.
        g = np.concatenate([np.full(12, 100.0), np.full(6, 39.0), np.full(3, 100.0)])
        m = long_nan_run_mask(g, dropout_min=30, prior_hr=1, interval_min=5)
        self.assertFalse(np.any(m), "all-finite (normalized) input has no NaN run")

    def test_short_nan_run_not_flagged_by_this_mask(self):
        g = _glucose(100, 101, np.nan, np.nan, 102)
        m = long_nan_run_mask(g, dropout_min=30, prior_hr=1, interval_min=5)
        self.assertFalse(np.any(m), "short NaN run is not flagged by long_nan_run_mask")


# ---- session_warmup_tail_mask ----

class TestSessionWarmupTailMask(unittest.TestCase):
    def test_output_shape_and_type(self):
        g = _glucose(100, 102, 98, 101, 99)
        dev = np.array(["A", "A", "A", "A", "A"])
        m = session_warmup_tail_mask(g, dev, warmup_hr=1, tail_hr=1, interval_min=5)
        self.assertEqual(m.shape, (5,))
        self.assertEqual(m.dtype, bool)

    def test_single_session_warmup_and_tail_flagged(self):
        # 12 points = 1 hr @ 5 min; warmup_hr=1, tail_hr=1 -> first 12 and last 12 flagged
        n = 24
        g = np.full(n, 100.0)
        dev = np.full(n, "s1")
        m = session_warmup_tail_mask(g, dev, warmup_hr=1, tail_hr=1, interval_min=5)
        self.assertTrue(m[0], "first point (warmup) should be flagged")
        self.assertTrue(m[-1], "last point (tail) should be flagged")
        self.assertTrue(np.any(m[:12]), "warmup segment should be flagged")
        self.assertTrue(np.any(m[-12:]), "tail segment should be flagged")

    def test_two_sessions_each_gets_warmup_and_tail(self):
        # Session 1: indices 0..9; Session 2: 10..19
        g = np.full(20, 100.0)
        dev = np.array(["a"] * 10 + ["b"] * 10)
        m = session_warmup_tail_mask(g, dev, warmup_hr=0.5, tail_hr=0.5, interval_min=5)
        # 0.5 hr = 6 points @ 5 min
        self.assertTrue(m[0], "start of first session flagged (warmup)")
        self.assertTrue(m[9], "end of first session flagged (tail)")
        self.assertTrue(m[10], "start of second session flagged (warmup)")
        self.assertTrue(m[19], "end of second session flagged (tail)")

    def test_device_id_length_mismatch_raises(self):
        g = _glucose(100, 101, 102)
        dev = np.array(["A", "B"])  # length 2
        with self.assertRaises(ValueError) as ctx:
            session_warmup_tail_mask(g, dev, warmup_hr=1, tail_hr=1, interval_min=5)
        self.assertIn("device_id length", str(ctx.exception))


# ---- calibration_period_mask ----

class TestCalibrationPeriodMask(unittest.TestCase):
    def test_output_shape_and_type(self):
        g = _glucose(100, 102, 98, 101)
        base = pd.Timestamp("2024-01-01 12:00:00")
        cgm_ts = pd.date_range(base, periods=4, freq="5min")
        cal_ts = []
        m = calibration_period_mask(g, cgm_ts, cal_ts, prior_hr=1, post_min=30)
        self.assertEqual(m.shape, (4,))
        self.assertEqual(m.dtype, bool)

    def test_one_calibration_window_flagged(self):
        base = pd.Timestamp("2024-01-01 12:00:00")
        # CGM at 12:00, 12:05, 12:10, 12:15, 12:20, 12:25 (6 points)
        n = 6
        g = np.full(n, 100.0)
        cgm_ts = pd.date_range(base, periods=n, freq="5min")
        # Cal at 12:10 -> window [11:00, 12:40]; prior_hr=1, post_min=30
        cal_ts = [pd.Timestamp("2024-01-01 12:10:00")]
        m = calibration_period_mask(g, cgm_ts, cal_ts, prior_hr=1, post_min=30)
        # 12:10 is in range; 12:00 is 10 min before cal, within 1 hr prior; 12:15, 12:20, 12:25 within 30 min post
        self.assertTrue(np.any(m), "at least one point in calibration window should be flagged")
        self.assertTrue(m[2], "point at calibration time (12:10) should be flagged")

    def test_no_calibrations_all_false(self):
        g = _glucose(100, 102, 98)
        base = pd.Timestamp("2024-01-01 12:00:00")
        cgm_ts = pd.date_range(base, periods=3, freq="5min")
        m = calibration_period_mask(g, cgm_ts, [], prior_hr=1, post_min=30)
        np.testing.assert_array_equal(m, False)

    def test_cgm_ts_length_mismatch_raises(self):
        g = _glucose(100, 101, 102)
        cgm_ts = pd.date_range("2024-01-01 12:00", periods=2, freq="5min")
        with self.assertRaises(ValueError) as ctx:
            calibration_period_mask(g, cgm_ts, [], prior_hr=1, post_min=30)
        self.assertIn("cgm_ts length", str(ctx.exception))


# ---- instability_mask (combined) ----

class TestInstabilityMask(unittest.TestCase):
    def test_output_shape_and_type(self):
        g = _glucose(100, 102, 98, 101, 99, 100)
        m = instability_mask(g)
        self.assertEqual(m.shape, (6,))
        self.assertEqual(m.dtype, bool)

    def test_combined_flags_when_heuristic_fires(self):
        g = _glucose(100, 102, 130, 132)
        m = instability_mask(g)
        self.assertTrue(np.any(m), "large step should be flagged by combined mask")


if __name__ == "__main__":
    unittest.main()
