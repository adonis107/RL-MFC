"""Build the report-specific figures that the standard plotting pipeline does not produce.

``scripts/plot.py`` writes one learning curve per benchmark on a common axis. On the
linear--quadratic benchmark the interesting part of that curve is the last few percent of
the range, which the full axis hides, so this script adds a zoomed panel and a plot of the
final optimality gap against the perturbation scale.

    uv run python scripts/report_figures.py --results-root results --output-root files/report/figures
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

from mfc.visualization import load_runs
from mfc.visualization.io import validation_dataframe


def lq_convergence(results_root, output_path, horizon=20, flow="exact", zoom_from=4000):
    runs = load_runs(results_root, env="lq")
    runs = [
        run
        for run in runs
        if run["metadata"]["horizon"] == horizon
        and (run["metadata"]["flow"] == flow or run["metadata"]["algorithm"] == "reinforce")
    ]
    curves = validation_dataframe(runs)
    objectives = pd.read_csv(Path(results_root) / "figures" / "lq" / "objectives.csv")
    objectives = objectives[(objectives["horizon"] == horizon) & (objectives["flow"] == flow)]
    optimum = float(objectives["J0_star"].dropna().iloc[0])

    figure, (left, right) = plt.subplots(1, 2, figsize=(11, 4.2))

    # Left: the tail of the learning curves, where the estimators actually separate.
    tail = curves[curves["step"] >= zoom_from]
    summary = tail.groupby(["label", "step"], as_index=False)["validation_reward"].agg(["mean", "std"]).reset_index()
    for label, group in summary.groupby("label", sort=False):
        group = group.sort_values("step")
        left.plot(group["step"], -group["mean"], label=label, linewidth=1.2)
        left.fill_between(
            group["step"], -group["mean"] - group["std"], -group["mean"] + group["std"], alpha=0.18
        )
    left.axhline(optimum, color="black", linestyle="--", linewidth=1.0, label="optimal cost")
    left.set_xlabel("training step")
    left.set_ylabel("validation cost (lower is better)")
    left.set_ylim(optimum - 0.02, optimum + 0.60)
    left.grid(alpha=0.25)
    left.legend(frameon=False, fontsize=7, loc="upper right", ncol=2)

    # Right: final gap to the optimum against the perturbation scale.
    transport = objectives[objectives["label"].str.startswith("Transport")].copy()
    transport["lambda"] = transport["label"].str.extract(r"lambda=([\d.]+)").astype(float)
    gaps = transport.groupby("lambda")["J0"].agg(["mean", "std"]).reset_index()
    gaps["gap"] = gaps["mean"] - optimum

    reinforce = objectives[objectives["label"] == "REINFORCE"]["J0"].mean() - optimum
    adaptive = objectives[objectives["label"].str.startswith("Adaptive")]["J0"].mean() - optimum

    right.plot(gaps["lambda"], gaps["gap"], marker="o", color="tab:blue", label="fixed-scale transport")
    right.axhline(reinforce, color="tab:orange", linestyle="-", linewidth=1.2, label="REINFORCE")
    right.axhline(adaptive, color="tab:green", linestyle=":", linewidth=1.4, label="adaptive transport")
    right.set_xscale("log")
    right.set_yscale("log")
    right.set_xticks(gaps["lambda"].to_numpy())
    right.set_xticklabels([f"{value:g}" for value in gaps["lambda"]])
    right.minorticks_off()
    right.set_xlabel(r"perturbation scale $\lambda$")
    right.set_ylabel(r"$J(\theta_\lambda)-J(\theta^\star)$")
    right.grid(alpha=0.25, which="both")
    right.legend(frameon=False, fontsize=8)

    figure.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, bbox_inches="tight", dpi=180)
    plt.close(figure)
    print(f"wrote {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Build report-specific figures from saved MFC runs.")
    parser.add_argument("--results-root", default=str(ROOT / "results"))
    parser.add_argument("--output-root", default=str(ROOT / "files" / "report" / "figures"))
    return parser.parse_args()


def main():
    args = parse_args()
    lq_convergence(args.results_root, Path(args.output_root) / "lq_convergence.png")


if __name__ == "__main__":
    main()
