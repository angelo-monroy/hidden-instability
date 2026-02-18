# CGM Governance Instability MVA

This project investigates how instability in consumer medical AI (CGM) propagates into downstream metrics and models under sparse validation. It illustrates how structural governance can effectuate blind spots without making overt clinical or population-level claims surrounding the biological data.

CGMs are biological sensors. They are FDA-approved medical devices, with their data used alone for treatment decisions. As consumers of this data (and as with most wearable data streams), we are subject to the system's disturbances that may not be called out by the system itself. This project uses algorithmic heuristics to mark periods that look unstable (noise, spikes, drift, dropouts), and create masks for downstream effect modeling.

Periods of hidden instability get a boolean mask (True = unstable). That mask is then used to:
- Exclude unstable segments when computing CGM metrics (TIR/TBR/TAR), and
- Compare metrics with vs without masking to show how much they depend on unstable segments.

As a person with Diabetes and long-term history with CGM, I can look at my current CGM data and understand a sensor's current instability intuitively. This project seeks to use CGM data as an exemplar case study for understanding how data is propogated across domains (the enterprise, the individual, the regulator, etc.). As a behavioral scientist at Dexcom, I'm familiar with glycemic analyses that assume CGM stream values are ground truth. This project challenges this assertion and simulates downstream effects via different sensitivities.

See [PROJECT_PLAN.md](PROJECT_PLAN.md) for constraints, data, and notebook roadmap.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Layout

- **`src/`** – Shared pipeline: instability heuristics → binary mask, TIR/TBR/TAR with optional masking. Python for data-heavy compute; usable from R via `reticulate` or from Python notebooks.
- **`data/`** – Raw and processed data (local only; not committed). Synthetic outputs go under `data/processed/` when notebooks added.
- **Notebooks** – To be added after recircle (R preferred: tidyverse, `|>`, gtsummary; Python acceptable; CGM metrics may use this Python core for speed/parallelism).

## Data

All raw data stays local. Only synthetic or derived artifacts may be committed. Do not commit `data/raw/`.

## Workflow

- `main`: stable, reproducible. `dev`: active development. `exp-*`: experiments.
- Commit before Cursor refactors. Semantic messages: `feat:`, `refactor:`, `exp:`.
