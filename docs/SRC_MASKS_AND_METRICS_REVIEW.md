# Review: `src` masks and CGM metrics

Walkthrough of every function so you can align with CGM conventions and add metrics. Your feedback will drive changes.

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

### 1.3 `drift_window_mask(glucose, *, drift_duration_hr=24, low_threshold_mgdL=70, low_duration_hr=8, interval_min=5)`

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

### 1.4 `dropout_flatline_mask(glucose, window_min=30, interval_min=5)`

**Intent:** Flag “dropout” or sensor dropout where the value is stuck (e.g. repeated identical readings).

**Logic:**
- Window = 30 min (6 readings).
- If all values in the window are **exactly equal** (and finite), the **entire window** is marked `True`.

**Choices you may want to change:**
- **Exact equality:** Real CGM can have tiny float differences. You might want “flatline” = “all within X mg/dL” (e.g. range in window ≤ 1 or 2 mg/dL) instead of strict equality.
- **Minimum duration:** 30 min is from the plan; some definitions use 15 min or 20 min.
- **Handling NaN:** Right now we require `np.all(np.isfinite(w))`; a window with any NaN is not marked as flatline. You might want “all non-NaN values in window are identical” and still flag.

---

### 1.5 `instability_mask(glucose, ...)`

**Intent:** Single combined “unstable” mask = union of all four heuristics.

**Logic:**
- Calls the four masks above with the given parameters.
- `instability_mask = local_variance | jump_spike | drift_window | dropout_flatline` (element-wise OR).
- Length is forced to match `glucose` (padding/trimming if a heuristic returns a different length).

**Choices you may want to change:**
- **Which heuristics are included:** You might want to turn off one (e.g. variance) or add new ones (e.g. “suspicious rate of change,” “out-of-physiologic-range”).
- **Parameters:** All are keyword-only and passed through; defaults can be updated from your conventions.

---

## 2. CGM metrics (`src/metrics.py`)

**Convention:** All metrics are **fractions of (used) time** in [0, 1]. If a `mask` is provided, masked indices are removed from **both** numerator and denominator (we compute metrics over “stable” time only when masking).

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

### 2.4 Relationship between TIR, TBR, TAR

- Currently: **TIR + TBR + TAR** = 1 (for every unmasked finite reading, exactly one of “in range,” “below,” “above”).
- Bounds: TIR uses [70, 180]; TBR uses <70; TAR uses >180. So no gap and no overlap.

---

## 3. Metrics not yet in `src`

Placeholder for ones you want to add. Common CGM metrics you might introduce:

- **GMI** (glucose management indicator) – e.g. formula from mean glucose.
- **CV** (coefficient of variation) – std / mean, often as %.
- **Mean glucose** (over unmasked / used time).
- **Median glucose.**
- **J-index, M-value, etc.** – other composite metrics.
- **Time in specific bands** – e.g. 70–140, 54–70, >250.
- **LBGI / HBGI** (low / high blood glucose indices) – risk indices from Clarke et al.
- **Event-based:** e.g. number or rate of excursions, prolonged hypoglycemia events (e.g. &gt;15 min below 54).

Once you specify which of these (and how you want them defined), they can be added to `src/metrics.py` with the same masking convention.

---

## 4. Summary table

| Function | Purpose | Key parameters | Your review |
|----------|---------|-----------------|-------------|
| `local_variance_mask` | High short-term variance | window 30 min, threshold 95th %ile | |
| `jump_spike_mask` | Single-step spike | 20 mg/dL per 5 min | |
| `drift_window_mask` | Monotonic > 24 h OR below 70 for > 8 h | drift_duration_hr=24, low=70, low_duration_hr=8 | |
| `dropout_flatline_mask` | Identical readings 30 min | exact equality, 30 min | |
| `instability_mask` | OR of all four | all of the above | |
| `compute_TIR` | % time in [70, 180] | low, high | |
| `compute_TBR` | % time < 70 | low | |
| `compute_TAR` | % time > 180 | high | |

---

**Next step:** Tell me what you want changed (definitions, thresholds, which indices get flagged, or new metrics), and I’ll update `src` and this doc accordingly.
