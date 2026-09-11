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

## Continuous-state transport

The continuous estimator represents the population law by a `K`-component Gaussian mixture fitted
to a block of population particles, perturbs the mixture coordinates, and corrects the policy
gradient by the sensitivity of those coordinates to the policy parameters. Set `K` with
`--n-components` (default 3). `--flow exact` is a single-Gaussian oracle that reads the population
flow from an environment's analytic moments instead of fitting it, and is only available with
`--n-components 1`.

The mixture is only identified when the population law really needs `K` components: otherwise the
likelihood is flat along a reparametrization of the chart, and `jacobian_floor` drops those
directions from the sensitivity solve (they are counted per update under `sensitivity_fallbacks`).
To choose `K` from measurements rather than by assumption, sweep it at a saved policy:

```bash
uv run python scripts/plot.py --env lq --results-root results --output-root results/figures --identification-replications 20
```

which writes `mixture_identification.csv`: the condition number of the score Jacobian, the share of
chart directions the floor retains, and the dispersion and bias of the estimator, for each `K` and
each floor.

Generate plots and diagnostic tables from a results directory:

```bash
uv run python scripts/plot.py --env all --results-root results --output-root results/plots
```

For low-SNR Portfolio gradient diagnostics, raise the diagnostic particle count without changing the training budget:

```bash
uv run python scripts/plot.py --env portfolio --results-root results --output-root results/plots --gradient-replications 20 --gradient-particles 8192
```

Generated experiment outputs are written under `results/` by default.
