# Project Plan: CGM Governance Instability MVA

## Purpose
This project investigates how instability in consumer medical AI systems propagates into downstream metrics and models under sparse validation conditions. The goal is to illustrate structural governance blind spots without making clinical or population-level claims.

## Core Constraints
- No continuous ground truth; BGM events are extremely sparse
- Validation signals are sporadic and user-initiated
- No population inference or cross-device comparisons
- No accuracy estimation or correction algorithms
- No policy prescriptions or clinical recommendations

## Available Data
1. **Personal CGM Data (Private)**  
   - Continuous 5-minute glucose readings  
   - Event metadata: sensor metrics (start/end), rare fingerstick BGMs, sporadic meals, missingness (time between sensors)
   - Usage: instability detection and temporal misalignment illustration  

2. **Synthetic CGM Data (Shareable)**  
   - Generated via Tidepool or equivalent  
   - Usage: error injection, validation sparsity simulation, downstream ML stress testing  

3. **Reddit Text Data (Public)**  
   - User posts mentioning sensor issues  
   - Usage: qualitative failure mode mapping  

## Derived vs Simulated

This section clarifies which aspects of the project come directly from real CGM data versus what is generated synthetically for experimentation.  

### Derived from Real CGM
All features in this category are calculated directly from your private CGM time series and do not rely on assumptions about true glucose values.

1. **Instability Heuristics**  
   - Defined algorithmically for each sensor session using the raw CGM time series:  
     - `local_variance`: rolling variance over 30-minute windows
       - Stable glucose doesn’t jump around much in short windows. If a 30‑min window has unusually high variance, that can indicate sensor noise, compression artifacts, or unreliable readings.
     - `jump_spike`: consecutive readings with absolute change > X mg/dL within 5 minutes  
     - `drift_window`: monotonic deviation over > 3 hours exceeding threshold  
     - `dropout_flatline`: repeated identical readings over > 30 minutes  
   - Output: a binary instability mask array, e.g. `instability_mask = [0,0,1,1,0,...]` aligned with timestamps.

2. **Metric Sensitivity Masks**  
   - Use `instability_mask` to calculate TIR, TBR, TAR metrics with and without unstable segments:  
     - `TIR_masked = compute_TIR(glucose_series, mask=instability_mask)`  
     - `TIR_unmasked = compute_TIR(glucose_series, mask=None)`  
   - This allows quantification of how much metrics fluctuate depending on which segments are included.

---

### Simulated
These are entirely generated or manipulated for stress-testing downstream metrics and models. They do **not use real BGM ground truth**.

1. **Error Injection**  
   - Apply simulated instability regimes to either synthetic CGM data or derived segments from real CGM:  
     - `inject_drift(series, start_idx, end_idx, slope)`  
     - `inject_spike(series, idx, magnitude)`  
     - `inject_flatline(series, start_idx, end_idx)`  

2. **Sparse Validation Events**  
   - Generate synthetic BGM or calibration points with realistic frequency:  
     - `validation_points = simulate_BGM(series, freq_days=14)`  
   - Align these randomly with synthetic or real CGM segments to model oversight gaps.

3. **Downstream ML Behavior**  
   - Train simple, interpretable models on synthetic or masked CGM:  
     - `model = train_model(input_series, target_labels)`  
   - Evaluate under:  
     - full observability (`performance_full`)  
     - sparse validation (`performance_sparse`)  
   - Compare results to demonstrate model brittleness under limited validation.

---

**Key Principle**  
- Derived = “observed directly from real data, used to define instability and metric sensitivity.”  
- Simulated = “engineered scenarios to stress-test how metrics and ML models behave when instability occurs and validation is sparse.”  
- These categories remain strictly separate to preserve reproducibility and prevent unintentional leakage of assumptions. 

## Notebooks
1. `01_instability_detection.ipynb` – detect instability regimes without ground truth  
2. `02_metric_fragility.ipynb` – compute glycemic metric (TIR/TB70/TA180) sensitivity under instability masks  
3. `03_sparse_validation_sim.ipynb` – simulate sparse validation and downstream ML impact  
4. `04_reddit_failure_modes.ipynb` – map user-reported issues to modeled instability regimes  

## Experiment Rules
- Commit before any Cursor refactor or code generation  
- Use `exp-*` branches for Cursor experiments  
- Keep ML models simple and interpretable  
- Limit total figures to ≤10  
- All raw data must remain local and never committed  

## Definition of Done
- All notebooks run end-to-end  
- Results reproducible with synthetic data  
- Clear demonstration of metric and model fragility under sparse validation  
- Limitations and assumptions explicitly documented  

## Git & Workflow Guidelines
- `main`: stable, fully reproducible work  
- `dev`: ongoing active development  
- `exp-*`: Cursor/AI experiments  
- Commit messages should be semantic (`feat:`, `refactor:`, `exp:`)  
- Frequent commits before invoking Cursor to allow instant rollback  

---

