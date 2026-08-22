import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "scripts" / "train.py"

DISCRETE_REFERENCE = {
    "twostate": {"n_particles": 200, "n_gradient": 10},
    "cybersecurity": {"n_particles": 200, "n_gradient": 1},
    "distribution": {"n_particles": 500, "n_gradient": 10},
    "advertising": {"n_particles": 200, "n_gradient": 10},
}

CONTINUOUS_REFERENCE = {
    "lq": {"n_particles": 200, "n_gradient": 1},
    "portfolio": {"n_particles": 500, "n_gradient": 1},
}


def job(env, algorithm, horizon, flow="exact", perturbation=None):
    return {
        "env": env,
        "algorithm": algorithm,
        "horizon": horizon,
        "flow": flow,
        "perturbation": perturbation,
    }


def experiment_plan(env):
    if env == "twostate":
        jobs = []
        for horizon in (2, 5):
            jobs.append(job(env, "reinforce", horizon))
            for flow in ("exact", "particle"):
                jobs.append(job(env, "mfreinforce", horizon, flow=flow, perturbation=0.2))
                for lambda_ in (0.05, 0.1, 0.2, 0.4, 0.8):
                    jobs.append(job(env, "transport", horizon, flow=flow, perturbation=lambda_))
        return jobs

    if env == "cybersecurity":
        jobs = [job(env, "reinforce", 3)]
        jobs.append(job(env, "mfreinforce", 3, perturbation=1.0))
        jobs.extend(job(env, "transport", 3, perturbation=lambda_) for lambda_ in (0.1, 0.2, 0.4))
        return jobs

    if env == "distribution":
        jobs = [job(env, "reinforce", 5)]
        jobs.append(job(env, "mfreinforce", 5, perturbation=2.0))
        jobs.extend(job(env, "transport", 5, perturbation=lambda_) for lambda_ in (0.1, 0.2, 0.4))
        return jobs

    if env == "advertising":
        jobs = [job(env, "reinforce", 5)]
        jobs.append(job(env, "mfreinforce", 5, perturbation=1.0))
        jobs.extend(job(env, "transport", 5, perturbation=lambda_) for lambda_ in (0.1, 0.2, 0.4))
        return jobs

    if env == "lq":
        jobs = [job(env, "reinforce", 20)]
        for flow in ("exact", "particle"):
            jobs.extend(job(env, "transport", 20, flow=flow, perturbation=lambda_) for lambda_ in (0.05, 0.1, 0.2, 0.4, 0.8))
        return jobs

    if env == "portfolio":
        jobs = [job(env, "reinforce", 10)]
        jobs.extend(job(env, "transport", 10, perturbation=lambda_) for lambda_ in (0.1, 0.2, 0.4))
        return jobs

    raise ValueError(f"Unknown environment: {env}")


def reference_budget(env):
    if env in DISCRETE_REFERENCE:
        return DISCRETE_REFERENCE[env]
    return CONTINUOUS_REFERENCE[env]


def mf_reference_cost(env, horizon):
    reference = reference_budget(env)
    particles = reference["n_particles"]
    gradient_samples = reference["n_gradient"]
    trajectory_cost = particles * horizon
    state_gradient_cost = 2 * gradient_samples * horizon * (horizon + 1) / 2.0
    return trajectory_cost + state_gradient_cost


def fair_run_parameters(job_spec):
    env = job_spec["env"]
    algorithm = job_spec["algorithm"]
    horizon = job_spec["horizon"]
    reference = reference_budget(env)
    ref_particles = reference["n_particles"]
    ref_gradient = reference["n_gradient"]
    base_cost = mf_reference_cost(env, horizon)

    parameters = {}
    if algorithm == "mfreinforce":
        parameters["n_particles"] = ref_particles
        parameters["n_logit_gradient"] = ref_gradient
    elif algorithm == "reinforce":
        parameters["n_particles"] = max(1, round(base_cost / horizon))
    elif algorithm == "transport":
        scale = base_cost / (horizon * (ref_particles + ref_gradient))
        parameters["n_particles"] = max(1, round(scale * ref_particles))
        if env in DISCRETE_REFERENCE:
            parameters["n_logit_gradient"] = max(1, round(scale * ref_gradient))
        else:
            parameters["n_law_gradient"] = max(1, round(scale * ref_gradient))

    if job_spec["flow"] == "particle" and algorithm in {"mfreinforce", "transport"}:
        parameters["n_flow_particles"] = ref_particles

    return parameters


def command_for(job_spec, seed, args):
    command = [
        sys.executable,
        str(TRAIN),
        "--env",
        job_spec["env"],
        "--algorithm",
        job_spec["algorithm"],
        "--horizon",
        str(job_spec["horizon"]),
        "--flow",
        job_spec["flow"],
        "--seed",
        str(seed),
        "--results-root",
        args.results_root,
    ]

    if job_spec["perturbation"] is not None:
        command.extend(["--perturbation", str(job_spec["perturbation"])])

    fair_parameters = fair_run_parameters(job_spec) if args.budget_mode == "fair" else {}

    optional_values = {
        "--device": args.device,
        "--n-train": args.n_train,
        "--lr": args.lr,
        "--n-particles": args.n_particles if args.n_particles is not None else fair_parameters.get("n_particles"),
        "--n-logit-gradient": args.n_logit_gradient
        if args.n_logit_gradient is not None
        else fair_parameters.get("n_logit_gradient"),
        "--n-law-gradient": args.n_law_gradient
        if args.n_law_gradient is not None
        else fair_parameters.get("n_law_gradient"),
        "--n-law-particles": args.n_law_particles,
        "--n-flow-particles": args.n_flow_particles
        if args.n_flow_particles is not None
        else fair_parameters.get("n_flow_particles"),
        "--validation-interval": args.validation_interval,
        "--simplex-sigma": args.simplex_sigma,
    }
    for flag, value in optional_values.items():
        if value is not None:
            command.extend([flag, str(value)])

    if args.law_chart is not None:
        command.extend(["--law-chart", args.law_chart])
    if args.baseline:
        command.append("--baseline")
    if args.no_baseline:
        command.append("--no-baseline")
    if args.no_reuse_state_gradient:
        command.append("--no-reuse-state-gradient")

    return command


def parse_seed_list(value):
    return [int(seed) for seed in value.split(",") if seed.strip()]


def parse_args():
    parser = argparse.ArgumentParser(description="Launch the training grid for one environment.")
    parser.add_argument(
        "--env",
        choices=["twostate", "cybersecurity", "distribution", "advertising", "lq", "portfolio", "all"],
        required=True,
    )
    parser.add_argument("--seeds", type=parse_seed_list, default=[0, 1, 2, 3, 4])
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--budget-mode", choices=["fair", "manual"], default="fair")
    parser.add_argument("--device", default=None)
    parser.add_argument("--n-train", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--n-particles", type=int, default=None)
    parser.add_argument("--n-logit-gradient", type=int, default=None)
    parser.add_argument("--n-law-gradient", type=int, default=None)
    parser.add_argument("--n-law-particles", type=int, default=None)
    parser.add_argument("--n-flow-particles", type=int, default=None)
    parser.add_argument("--validation-interval", type=int, default=None)
    parser.add_argument("--simplex-sigma", type=float, default=None)
    parser.add_argument("--law-chart", choices=["gaussian", "mean"], default=None)
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--no-baseline", action="store_true")
    parser.add_argument("--no-reuse-state-gradient", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.baseline and args.no_baseline:
        raise ValueError("Use at most one of --baseline and --no-baseline.")

    envs = ["twostate", "cybersecurity", "distribution", "advertising", "lq", "portfolio"]
    selected_envs = envs if args.env == "all" else [args.env]

    commands = []
    for env in selected_envs:
        for job_spec in experiment_plan(env):
            for seed in args.seeds:
                commands.append(command_for(job_spec, seed, args))

    print(f"Prepared {len(commands)} training jobs.")
    for command in commands:
        print(" ".join(command))

    if args.dry_run:
        return

    for index, command in enumerate(commands, start=1):
        print(f"\n[{index}/{len(commands)}] {' '.join(command)}", flush=True)
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
