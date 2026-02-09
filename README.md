# CGM Governance Instability MVA

Investigates how instability in consumer medical AI (CGM) propagates into downstream metrics and models under sparse validation. Illustrates structural governance blind spots without clinical or population-level claims.

See [PROJECT_PLAN.md](PROJECT_PLAN.md) for purpose, constraints, data, and notebook roadmap.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Layout

- **`src/`** – Shared pipeline: instability heuristics → binary mask, TIR/TBR/TAR with optional masking. Python for data-heavy compute; usable from R via `reticulate` or from Python notebooks.
- **`data/`** – Raw and processed data (local only; not committed). Synthetic outputs go under `data/processed/` when we add notebooks.
- **Notebooks** – To be added after recircle (R preferred: tidyverse, `|>`, gtsummary; Python acceptable; CGM metrics may use this Python core for speed/parallelism).

## Data

All raw data stays local. Only synthetic or derived artifacts may be committed. Do not commit `data/raw/`.

## Workflow

- `main`: stable, reproducible. `dev`: active development. `exp-*`: experiments.
- Commit before Cursor refactors. Semantic messages: `feat:`, `refactor:`, `exp:`.
