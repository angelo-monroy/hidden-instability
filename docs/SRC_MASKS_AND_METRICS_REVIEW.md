# Review: `src` masks and CGM metrics

Walkthrough of every function

---

## 1. Instability masks (`src/instability.py`)

**Convention:** All masks are boolean arrays aligned to the glucose time series. `True` = unstable (excluded when computing masked metrics).

---

### 1.1 `local_variance_mask(glucose, window_min=30, interval_min=5, threshold=None)`

**Intent:** Flag periods where glucose is “noisy” over a short window (sensor noise, compression, etc.).

**Logic:**
- Assumes 5‑min interval: `window_min=30` → 6 readings per window.
- At each time index `i`, compute variance over `[i − 5, …, i]` (30 min ending at current time).
- If `threshold` is `None`: threshold = 95th percentile of those rolling variances (adaptive).
- Mask is `True` only at the **right edge** of each window (the “current” time), not the whole window.

**Choices you may want to change:**
- **Window:** 30 min fixed; could be parameterized or different (e.g. 15 min).
- **Threshold:** 95th percentile is arbitrary; you may prefer a fixed mg/dL² or device-specific cutoffs.
- **Which indices are flagged:** Only the last index of the window is `True`. If you want “flag the entire 30‑min window when variance is high,” that would require a different rule (e.g. broadcast to all indices in that window).

---

### 1.2 `jump_spike_mask(glucose, threshold_mgdL=20, interval_min=5)`

**Intent:** Flag single-step “spikes” (e.g. sensor artifact, transient misread).

**Logic:**
- `diff = |glucose[i] − glucose[i−1]|` (one 5‑min step).
- Mask is `True` wherever `diff > threshold_mgdL` (default 20 mg/dL).

**Choices you may want to change:**
- **Threshold:** 20 mg/dL is a common rule-of-thumb; you may want different (e.g. 15, 25) or ADA/consensus if one exists.
- **Step:** Exactly one interval; no “change over 2 or 3 steps” variant yet.
- **Which reading is flagged:** Both the “before” and “after” readings could be considered suspect; currently we flag the index where the **jump lands** (the later reading). Could instead flag both, or only the “peak” side.

---

### 1.3 `jitter_mask(glucose, window_min=30, min_sign_changes=2, interval_min=5)`

**Intent:** Flag periods with **too much small oscillation (jitter)**—many direction reversals even when individual steps are small (e.g. 5–10 mg/dL up and down). Good CGM looks like a sequential line; jittery data oscillates around an imaginary average with no smooth trend (scatter vs. dotted line).

**Logic:** In each rolling window (default 30 min = 6 points), compute first differences and count **sign changes** (direction reversals: diff[i]*diff[i+1] &lt; 0). If sign changes ≥ min_sign_changes (default 2), mark the **entire window** `True`. Requires all finite in window.

**Choices you may want to change:** window_min (30 min), min_sign_changes (2), or use a ratio (e.g. sign changes per minute) instead of raw count.

---

### 1.4 `drift_window_mask(glucose, *, drift_duration_hr=24, low_threshold_mgdL=70, low_duration_hr=8, interval_min=5)`

**Intent:** Flag two patterns that may indicate sensor drift or governance blind spots (not normal physiologic swings from insulin/carbs):

1. **Monotonic drift > 24 h:** Glucose drifts in one direction for longer than a day. Possible sensor drift or untreated elevation; flagged as potential instability, not verified.
2. **Below range for > 8 h:** Glucose is below 70 mg/dL and stays there for more than 8 hours. Flagged as potential sensor drift (reading low) or prolonged hypo.

**Logic:**
- **Monotonic:** Sliding window of 24 h (288 points @ 5 min). For each window, require all finite; if `np.diff(window)` is all ≥ 0 or all ≤ 0, mark the **entire window** `True`.
- **Below 70 for 8+ h:** Find contiguous runs where `glucose < 70` (and finite). If run length > 96 points (8 h), mark **all indices in that run** `True`.

**Cost:** Monotonic check is O(n × 288) for 24 h windows; low run is O(n). For 30 days @ 5 min, ~2.5e6 ops for monotonic, negligible for low run.

**Choices you may want to change:**
- **Drift duration:** 24 h default; could be 12 h or 48 h.
- **Low threshold:** 70 mg/dL (below range); could add level 2 (e.g. 54).
- **Low duration:** 8 h default; could be 4 h or 12 h.
- **Monotonic strictness:** Currently strict (every step same sign); could relax with a tolerance for small reversals.

---

### 1.5 `dropout_flatline_mask(glucose, window_min=30, interval_min=5)`

**Intent:** Flag two patterns:

1. **Flatline:** Repeated identical readings for 30+ min (sensor stuck on one value).
2. **Dropout:** **Any** reading that is NaN. The occasional NaN between two numeric values is still dropout and is flagged. Every NaN is marked unstable.

**Logic:**
- **Flatline:** Window = 30 min (6 readings). If all values in the window are **exactly equal** and finite, the **entire window** is marked `True`.
- **Dropout:** `mask = mask | ~np.isfinite(glucose)` so every NaN (and inf) is marked `True`.

**Session context:** Max session length depends on source device ID—see `max_session_days(device_id)` in `src.session`: **G7 → 10.5 days**, **G6 (and not G7) → 10 days**.

**Choices you may want to change:**
- **Flatline—exact equality:** You might want “flatline” = “all within X mg/dL” (e.g. range ≤ 1 or 2 mg/dL) instead of strict equality.
- **Flatline—minimum duration:** 30 min default; could be 15 or 20 min.

---

### 1.6 `long_nan_run_mask(glucose, dropout_min=30, prior_hr=1, interval_min=5)`

**Intent:** **Separate mask** for long NaN runs and the lead-up. When a NaN run is at least **dropout_min** (default 30 min), flag the entire run and the **prior_hr** (default 1 hour) **before** the run starts. Short NaN runs are not flagged by this mask—use `dropout_flatline_mask` for any NaN.

**Logic:** Find contiguous NaN runs. If run length ≥ dropout_min (e.g. 6 points = 30 min @ 5 min), mark the **entire run** and the **prior_hr** (e.g. 12 points = 1 hr) before the run start as `True`.

---

### 1.6 `instability_mask(glucose, ...)`

**Intent:** Single combined “unstable” mask = union of all six heuristics.

**Logic:**
- Calls the six masks above with the given parameters.
- `instability_mask = local_variance | jump_spike | jitter | drift_window | dropout_flatline | long_nan_run` (element-wise OR).
- Length is forced to match `glucose` (padding/trimming if a heuristic returns a different length).

**Choices you may want to change:**
- **Which heuristics are included:** You might want to turn off one (e.g. variance) or add new ones (e.g. “suspicious rate of change,” “out-of-physiologic-range”).
- **Parameters:** All are keyword-only and passed through; defaults can be updated from your conventions.

---

### 1.8 Session helper: `max_session_days(device_id)` (`src/session.py`)

**Intent:** Return maximum expected session length in days for a given source device ID. Use when flagging sessions that ended early (potential failure).

**Logic:**
- If `device_id` contains **"G7"** → **10.5 days**.
- If `device_id` contains **"G6"** and not **"G7"** → **10 days**.
- Otherwise → **None** (unknown device). Comparison is case-insensitive.

**Usage:** When grouping by CGM session ID, compare actual session duration to `max_session_days(device_id)`; if duration < max, flag as possible early failure.

---

## 2. CGM metrics (`src/metrics.py`)

**Convention:** TIR, TBR, and TAR are **fractions of (used) time** in [0, 1]. GMI is a **scalar** (estimated A1C-equivalent in %). Summary metrics return a **dict** of scalars (mean, SD, CV, median, min, max). If a `mask` is provided, masked indices are removed from computation (we compute metrics over “stable” time only when masking).

---

### 2.1 `compute_TIR(glucose, mask=None, low=70, high=180)`

**Intent:** Time in range [low, high] as fraction of total time.

**Logic:**
- Count indices where `low ≤ glucose ≤ high` (and not masked, and finite).
- Denominator = count of all (unmasked, finite) indices.
- Return that fraction.

**Choices you may want to change:**
- **Bounds:** Default 70–180 mg/dL (ADA-style). You may want 70–140 (tight), or different bounds (e.g. 63–140 for some guidelines), or multiple TIR bands.
- **Denominator:** Currently “all unmasked finite readings”; alternative is “total time” if you have timestamps and want to weight by actual time (e.g. if interval is not exactly 5 min everywhere).

---

### 2.2 `compute_TBR(glucose, mask=None, low=70)`

**Intent:** Time below range (< low) as fraction of time.

**Logic:**
- Count where `glucose < low` (unmasked, finite).
- Denominator = same as TIR (all unmasked finite).

**Choices you may want to change:**
- **Threshold:** 70 mg/dL default; some use 54 for “level 2” hypoglycemia; you may want multiple bands (e.g. TBR &lt;54, TBR &lt;70).
- **Inclusive vs exclusive:** Currently strictly `< low`; if you want “below range” to be “≤ 69” that’s equivalent for integer values but worth stating.

---

### 2.3 `compute_TAR(glucose, mask=None, high=180)`

**Intent:** Time above range (> high) as fraction of time.

**Logic:**
- Count where `glucose > high` (unmasked, finite).
- Denominator = same as TIR/TBR.

**Choices you may want to change:**
- **Threshold:** 180 mg/dL default; some use 250 for “level 2” hyperglycemia; you may want multiple bands (e.g. TAR 180–250, TAR >250).
- **Inclusive vs exclusive:** Currently strictly `> high`.

---

### 2.4 `compute_GMI(glucose, mask=None)`

**Intent:** Glucose Management Indicator: estimated A1C-equivalent (%) from mean glucose (mg/dL). If a mask is provided, masked indices are omitted; mean is over used, finite values only.

**Logic:**
- Use `_masked_series(glucose, mask)` and restrict to **finite and used** indices (`valid = np.isfinite(arr) & use`).
- If no valid values, return `np.nan`.
- Otherwise: **mean_glucose** = mean of `arr[valid]` (mg/dL). **GMI (%)** = 3.31 + 0.02392 × mean_glucose (Bergenstal et al. formula).
- Returns a single float (percent, e.g. 7.0 for 7%).

**Choices you may want to change:**
- **Formula:** Default is 3.31 + 0.02392 × mean (mg/dL). Alternative formulas exist (e.g. different coefficients or mmol/L); you could add a parameter or switch if needed.
- **Units:** Input is assumed mg/dL; if you use mmol/L, the formula coefficients differ.

---

### 2.5 `compute_summary_metrics(glucose, mask=None)`

**Intent:** Summary statistics over used, finite glucose values: mean, SD, CV, median, min, max. If a mask is provided, masked indices are omitted (same convention as TIR/TBR/TAR).

**Logic:**
- Use `_masked_series(glucose, mask)` to get `arr` and `use`.
- Restrict to **finite and used** indices (`valid = np.isfinite(arr) & use`).
- If no valid values, return a dict with all keys set to `np.nan`.
- Otherwise compute on `arr[valid]`:
  - **mean:** `np.mean(x)`
  - **sd:** `np.std(x, ddof=1)` (sample standard deviation)
  - **cv:** `sd / mean` (coefficient of variation as ratio). If mean is 0, returns `np.nan`.
  - **median:** `np.median(x)`
  - **min:** `np.min(x)`
  - **max:** `np.max(x)`
- Returns a **dict** with keys: `mean`, `sd`, `cv`, `median`, `min`, `max`.

**Choices you may want to change:**
- **CV:** Currently ratio (SD/mean); you may want CV% (e.g. 100 × SD/mean) as an additional key or parameter.
- **SD:** Currently sample SD (`ddof=1`); you may prefer population SD (`ddof=0`) for reporting.
- **Return type:** Dict is easy to use from Python and R (reticulate); could add a pandas Series or named tuple variant if needed.

---

### 2.6 Relationship between TIR, TBR, TAR

- Currently: **TIR + TBR + TAR** = 1 (for every unmasked finite reading, exactly one of “in range,” “below,” “above”).
- Bounds: TIR uses [70, 180]; TBR uses <70; TAR uses >180. So no gap and no overlap.

---

## 3. Metrics not yet in `src`

**Already in `src`:** TIR, TBR, TAR, **GMI** (via `compute_GMI`), and **summary metrics** (mean, SD, CV, median, min, max) via `compute_summary_metrics`.

- **J-index, M-value, etc.** – other composite metrics.
- **LBGI / HBGI** (low / high blood glucose indices) – risk indices from Clarke et al.
- **Event-based:** e.g. number or rate of excursions or spikes, prolonged hypoglycemia events (e.g. &gt;15 min below 54).

---

## 4. Summary table

| Function | Purpose | Key parameters | Your review |
|----------|---------|-----------------|-------------|
| `local_variance_mask` | High short-term variance | window 30 min, threshold 95th %ile | |
| `jump_spike_mask` | Single-step spike | 20 mg/dL per 5 min | |
| `jitter_mask` | Too many direction reversals (small oscillation) | window_min=30, min_sign_changes=2 | |
| `drift_window_mask` | Monotonic > 24 h OR below 70 for > 8 h | drift_duration_hr=24, low=70, low_duration_hr=8 | |
| `dropout_flatline_mask` | Flatline 30+ min OR any NaN (dropout) | window_min=30 | |
| `long_nan_run_mask` | Long NaN run (≥30 min) + 1 hr prior | dropout_min=30, prior_hr=1 | |
| `instability_mask` | OR of all six | all of the above | |
| `compute_TIR` | % time in [70, 180] | low, high | |
| `compute_TBR` | % time < 70 | low | |
| `compute_TAR` | % time > 180 | high | |
| `compute_GMI` | Estimated A1C-equivalent (%) from mean glucose | mask | |
| `compute_summary_metrics` | mean, SD, CV, median, min, max (dict) | mask | |

---
