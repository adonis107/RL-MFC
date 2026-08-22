# MFC

Mean-field control experiments comparing REINFORCE, MF-REINFORCE, and transport-gradient estimators across discrete and continuous benchmark environments.

## Layout

- `src/mfc/environments/`: benchmark environments such as TwoState, Advertising, LQ, and Portfolio.
- `src/mfc/algorithms/`: estimator and training implementations.
- `src/mfc/visualization/`: result loading, plots, objective tables, and gradient diagnostics.
- `scripts/train.py`: run one training job.
- `scripts/run.py`: launch experiment grids.
- `scripts/plot.py`: build CSV summaries and plots from saved runs.

## Quick Start

Run a small training job:

```bash
uv run python scripts/train.py --env lq --algorithm transport --horizon 20 --perturbation 0.2 --n-train 10 --device cpu
```

Generate plots and diagnostic tables from a results directory:

```bash
uv run python scripts/plot.py --env all --results-root results --output-root results/plots
```

Generated experiment outputs are written under `results/` by default.
