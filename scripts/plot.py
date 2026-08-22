import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mfc.visualization import (  # noqa: E402
    discrete_transport_tv_bound_table,
    gradient_diagnostics,
    best_runs_by_label,
    load_runs,
    objective_table,
    plot_advertising_diagnostics,
    plot_distribution_comparison,
    plot_flow_comparison,
    plot_state_flow,
    plot_validation_rewards,
    runtime_table,
    save_table,
    transport_correction_table,
    twostate_policy_error_table,
)


def save_current(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, bbox_inches="tight", dpi=180)
    plt.close()


def make_standard_outputs(
    env,
    results_root,
    output_root,
    gradient_replications=0,
    correction_replications=0,
    gradient_particles=None,
    correction_particles=None,
    allow_empty=False,
):
    runs = load_runs(results_root, env=env)
    if not runs:
        message = f"No runs found under {Path(results_root) / env}."
        if allow_empty:
            print(f"warning: {message}", file=sys.stderr)
            return False
        raise ValueError(message)

    output_dir = Path(output_root) / env
    output_dir.mkdir(parents=True, exist_ok=True)

    for horizon in sorted({run["metadata"]["horizon"] for run in runs}):
        horizon_runs = [run for run in runs if run["metadata"]["horizon"] == horizon]
        plot_validation_rewards(horizon_runs, env=env, horizon=horizon)
        save_current(output_dir / f"validation_T_{horizon}.png")

        transport_runs = [run for run in horizon_runs if run["metadata"]["algorithm"] == "transport"]
        flows = {run["metadata"]["flow"] for run in transport_runs}
        if "exact" in flows and "particle" in flows:
            plot_flow_comparison(transport_runs, env=env, horizon=horizon)
            save_current(output_dir / f"flow_comparison_T_{horizon}.png")

    save_table(runtime_table(runs), output_dir / "runtime.csv")
    save_table(objective_table(runs), output_dir / "objectives.csv")

    tv_table = discrete_transport_tv_bound_table(runs)
    if not tv_table.empty:
        save_table(tv_table, output_dir / "transport_tv_bounds.csv")

    if env == "twostate":
        save_table(twostate_policy_error_table(runs), output_dir / "policy_error.csv")

    for run in best_runs_by_label(runs):
        metadata = run["metadata"]
        stem = (
            f"{metadata['algorithm']}_"
            f"{metadata['perturbation'] if metadata['perturbation'] is not None else 'none'}_"
            f"T_{metadata['horizon']}_{metadata['flow']}"
        )
        if env == "distribution":
            plot_distribution_comparison(run)
            save_current(output_dir / f"distribution_{stem}.png")
        elif env == "advertising":
            plot_advertising_diagnostics(run)
            save_current(output_dir / f"advertising_{stem}.png")
        else:
            plot_state_flow(run)
            save_current(output_dir / f"state_flow_{stem}.png")

    if gradient_replications > 0:
        rows = []
        for run in runs:
            try:
                rows.append(
                    gradient_diagnostics(
                        run,
                        n_replications=gradient_replications,
                        n_particles=gradient_particles,
                    )
                )
            except ValueError:
                continue
        if rows:
            save_table(pd.concat(rows, ignore_index=True), output_dir / "gradient_diagnostics.csv")

    if correction_replications > 0:
        rows = []
        for run in runs:
            try:
                rows.append(
                    transport_correction_table(
                        run,
                        n_replications=correction_replications,
                        n_particles=correction_particles,
                    )
                )
            except ValueError:
                continue
        if rows:
            save_table(pd.concat(rows, ignore_index=True), output_dir / "transport_correction.csv")
    return True


def parse_args():
    parser = argparse.ArgumentParser(description="Create standard plots and tables from saved MFC runs.")
    parser.add_argument(
        "--env",
        choices=["twostate", "cybersecurity", "distribution", "advertising", "lq", "portfolio", "all"],
        required=True,
    )
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--output-root", default="results/figures")
    parser.add_argument("--gradient-replications", type=int, default=0)
    parser.add_argument("--correction-replications", type=int, default=0)
    parser.add_argument("--gradient-particles", type=int, default=None)
    parser.add_argument("--correction-particles", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    envs = ["twostate", "cybersecurity", "distribution", "advertising", "lq", "portfolio"]
    selected_envs = envs if args.env == "all" else [args.env]
    for env in selected_envs:
        make_standard_outputs(
            env,
            args.results_root,
            args.output_root,
            args.gradient_replications,
            args.correction_replications,
            args.gradient_particles,
            args.correction_particles,
            allow_empty=args.env == "all",
        )


if __name__ == "__main__":
    main()
