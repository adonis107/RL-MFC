import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

from mfc.visualization import (
    adaptive_schedule_dataframe,
    advertising_policy_error_table,
    discrete_transport_tv_bound_table,
    gradient_diagnostics,
    best_runs_by_label,
    flow_dataframe,
    load_runs,
    objective_table,
    plot_adaptive_schedule,
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


def perturbation_stem(metadata):
    perturbation = metadata["perturbation"] if metadata["perturbation"] is not None else "none"
    if metadata["algorithm"] not in {"transport", "adaptive_transport"}:
        return str(perturbation)

    eta = metadata.get("algorithm_config", {}).get("eta", metadata.get("eta"))
    if eta is None:
        return str(perturbation)
    return f"{perturbation}_eta_{eta}"


def value_stem(value):
    if value is None:
        return "none"
    if isinstance(value, float):
        return f"{value:g}".replace("-", "m").replace(".", "p")
    return str(value).replace("-", "m").replace(".", "p")


def sort_key(value):
    if value is None:
        return float("-inf")
    return value


def run_score(run):
    summary = run.get("summary", {})
    value = summary.get("last_validation_objective")
    if value is None:
        value = summary.get("last_objective")
    return float("-inf") if value is None else value


def best_transport_eta_runs(transport_runs):
    """Keep the eta with the best mean final validation for each lambda/flow."""
    selected = []
    group_keys = sorted(
        {
            (run["metadata"].get("perturbation"), run["metadata"].get("flow"))
            for run in transport_runs
        },
        key=lambda item: (sort_key(item[0]), str(item[1])),
    )

    for perturbation, flow in group_keys:
        group_runs = [
            run
            for run in transport_runs
            if run["metadata"].get("perturbation") == perturbation
            and run["metadata"].get("flow") == flow
        ]
        eta_values = sorted({run["metadata"].get("eta") for run in group_runs}, key=sort_key)
        if not eta_values:
            continue

        best_eta = max(
            eta_values,
            key=lambda eta: sum(
                run_score(run)
                for run in group_runs
                if run["metadata"].get("eta") == eta
            )
            / max(1, sum(1 for run in group_runs if run["metadata"].get("eta") == eta)),
        )
        selected.extend(run for run in group_runs if run["metadata"].get("eta") == best_eta)

    return selected


def validation_overview_runs(horizon_runs, flow=None):
    core_runs = []
    transport_runs = []
    for run in horizon_runs:
        metadata = run["metadata"]
        if flow is not None and metadata["algorithm"] != "reinforce" and metadata["flow"] != flow:
            continue
        if metadata["algorithm"] == "transport":
            transport_runs.append(run)
        else:
            core_runs.append(run)
    return core_runs + best_transport_eta_runs(transport_runs)


def representative_plot_runs(runs):
    core_runs = [run for run in runs if run["metadata"]["algorithm"] != "transport"]
    transport_runs = [run for run in runs if run["metadata"]["algorithm"] == "transport"]
    return best_runs_by_label(core_runs + best_transport_eta_runs(transport_runs))


def save_validation_splits(horizon_runs, env, horizon, output_dir):
    transport_runs = [run for run in horizon_runs if run["metadata"]["algorithm"] == "transport"]

    if not transport_runs:
        plot_validation_rewards(horizon_runs, env=env, horizon=horizon)
        save_current(output_dir / f"validation_T_{horizon}.png")
        return

    transport_flows = sorted({run["metadata"]["flow"] for run in transport_runs})
    default_flow = "exact" if "exact" in transport_flows else transport_flows[0]

    for flow in transport_flows:
        overview_runs = validation_overview_runs(horizon_runs, flow=flow)
        if not overview_runs:
            continue
        plot_validation_rewards(overview_runs, env=env, horizon=horizon)
        suffix = "" if flow == default_flow else f"_{flow}"
        save_current(output_dir / f"validation_T_{horizon}{suffix}.png")

    for flow in sorted({run["metadata"]["flow"] for run in transport_runs}):
        flow_runs = [run for run in transport_runs if run["metadata"]["flow"] == flow]
        for perturbation in sorted({run["metadata"].get("perturbation") for run in flow_runs}, key=sort_key):
            split_runs = [run for run in flow_runs if run["metadata"].get("perturbation") == perturbation]
            if not split_runs:
                continue
            plot_validation_rewards(split_runs, env=env, horizon=horizon)
            save_current(
                output_dir
                / f"validation_transport_eta_sweep_T_{horizon}_{flow}_lambda_{value_stem(perturbation)}.png"
            )


def save_flow_comparison_splits(transport_runs, env, horizon, output_dir):
    flows = {run["metadata"]["flow"] for run in transport_runs}
    if not {"exact", "particle"}.issubset(flows):
        return

    best_runs = best_transport_eta_runs(transport_runs)
    for perturbation in sorted({run["metadata"].get("perturbation") for run in best_runs}, key=sort_key):
        split_runs = [run for run in best_runs if run["metadata"].get("perturbation") == perturbation]
        split_flows = {run["metadata"]["flow"] for run in split_runs}
        if not {"exact", "particle"}.issubset(split_flows):
            continue
        plot_flow_comparison(split_runs, env=env, horizon=horizon)
        save_current(
            output_dir
            / (
                f"flow_comparison_T_{horizon}_"
                f"lambda_{value_stem(perturbation)}_best_eta.png"
            )
        )


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
        save_validation_splits(horizon_runs, env, horizon, output_dir)

        transport_runs = [run for run in horizon_runs if run["metadata"]["algorithm"] == "transport"]
        save_flow_comparison_splits(transport_runs, env, horizon, output_dir)

    save_table(runtime_table(runs), output_dir / "runtime.csv")
    save_table(objective_table(runs), output_dir / "objectives.csv")

    tv_table = discrete_transport_tv_bound_table(runs)
    if not tv_table.empty:
        save_table(tv_table, output_dir / "transport_tv_bounds.csv")

    if env == "twostate":
        save_table(twostate_policy_error_table(runs), output_dir / "policy_error.csv")
    if env == "advertising":
        save_table(advertising_policy_error_table(runs), output_dir / "policy_error.csv")

    if env in {"lq", "portfolio", "kuramoto"}:
        rows = []
        for run in representative_plot_runs(runs):
            metadata = run["metadata"]
            table = flow_dataframe(run)
            table.insert(0, "seed", metadata["seed"])
            table.insert(0, "flow", metadata["flow"])
            table.insert(0, "horizon", metadata["horizon"])
            table.insert(0, "eta", metadata.get("eta"))
            table.insert(0, "perturbation", metadata["perturbation"])
            table.insert(0, "label", (
                f"{metadata['algorithm']}_"
                f"{perturbation_stem(metadata)}_"
                f"{metadata['flow']}"
            ))
            rows.append(table)
        if rows:
            save_table(pd.concat(rows, ignore_index=True), output_dir / "moment_flows.csv")

    adaptive = adaptive_schedule_dataframe(runs)
    if not adaptive.empty:
        save_table(adaptive, output_dir / "adaptive_schedule.csv")
        for (horizon, flow), _ in adaptive.groupby(["horizon", "flow"]):
            plot_adaptive_schedule(
                runs,
                env=env,
                horizon=horizon,
                flow=flow,
                save_path=output_dir / f"adaptive_schedule_T_{horizon}_{flow}.png",
            )

    for run in representative_plot_runs(runs):
        metadata = run["metadata"]
        stem = (
            f"{metadata['algorithm']}_"
            f"{perturbation_stem(metadata)}_"
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
        choices=["twostate", "cybersecurity", "distribution", "advertising", "lq", "kuramoto", "portfolio", "all"],
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
    envs = ["twostate", "cybersecurity", "distribution", "advertising", "lq", "kuramoto", "portfolio"]
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
