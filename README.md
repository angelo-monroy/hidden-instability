# CGM Governance Instability MVA

Investigates how instability in consumer medical AI (CGM) propagates into downstream metrics and models under sparse validation. Illustrates structural governance blind spots without clinical or population-level claims.

CGM data has no continuous ground truth. We can’t label “true” vs “wrong” readings as consumers of this data, as with most wearable data streams. This project uses algorithmic heuristics to mark periods that look unstable (noise, spikes, drift, dropouts). Those periods get a boolean mask (True = unstable).

That mask is then used to:
- Exclude unstable segments when computing CGM metrics (TIR/TBR/TAR), and
- Compare metrics with vs without masking to show how much they depend on unstable segments.

As a person with Diabetes and long-term history with CGM, I can look at a CGM graph and understand this instability intuitively. Quantifying this however is necessary to understand how data is propogated across domains (the enterprise, the individual, the regulator, etc.). As a behavioral scientist at Dexcom, we performed glycemic analyses with the assumption that CGM stream values were ground truth, given the algorithmic engineering work done to produce the device in the first place. This project challenges this assertion and simulates downstream effects via different sensitivities.

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
