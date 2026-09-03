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
    "kuramoto": {"n_particles": 500, "n_gradient": 20},
    "portfolio": {"n_particles": 500, "n_gradient": 1},
}

TRANSPORT_LAMBDAS = (0.05, 0.1, 0.2, 0.4, 0.8)
TWOSTATE_TRANSPORT_ETAS = (0.4, 0.6, 0.85, 0.95)
DEFAULT_TRANSPORT_ETA = 0.85
ADAPTIVE_INITIAL_LAMBDA = 0.2
ADAPTIVE_INITIAL_ETA = 0.8
ADAPTIVE_CHECKPOINT_INTERVAL = 100
ADAPTIVE_REPLICATIONS = 4
# Reallocate a fixed transport budget toward the auxiliary sensitivity estimate
# where n=1 is too noisy; trajectory particles are reduced to keep cost equal.
TRANSPORT_AUXILIARY_GRADIENTS = {
    "cybersecurity": 20,
    "distribution": 64,
    "kuramoto": 50,
    "lq": 20,
    "portfolio": 200,
}


def job(env, algorithm, horizon, flow="exact", perturbation=None, eta=None):
    return {
        "env": env,
        "algorithm": algorithm,
        "horizon": horizon,
        "flow": flow,
        "perturbation": perturbation,
        "eta": eta,
    }


def experiment_plan(env):
    if env == "twostate":
        jobs = []
        for horizon in (2, 5):
            jobs.append(job(env, "reinforce", horizon))
            for flow in ("exact", "particle"):
                jobs.append(job(env, "mfreinforce", horizon, flow=flow, perturbation=0.2))
                for lambda_ in TRANSPORT_LAMBDAS:
                    for eta in TWOSTATE_TRANSPORT_ETAS:
                        jobs.append(job(env, "transport", horizon, flow=flow, perturbation=lambda_, eta=eta))
                jobs.append(
                    job(
                        env,
                        "adaptive_transport",
                        horizon,
                        flow=flow,
                        perturbation=ADAPTIVE_INITIAL_LAMBDA,
                        eta=ADAPTIVE_INITIAL_ETA,
                    )
                )
        return jobs

    if env == "cybersecurity":
        jobs = [job(env, "reinforce", 3)]
        jobs.append(job(env, "mfreinforce", 3, perturbation=1.0))
        jobs.extend(
            job(env, "transport", 3, perturbation=lambda_, eta=DEFAULT_TRANSPORT_ETA)
            for lambda_ in (0.1, 0.2, 0.4)
        )
        jobs.append(job(env, "adaptive_transport", 3, perturbation=ADAPTIVE_INITIAL_LAMBDA, eta=ADAPTIVE_INITIAL_ETA))
        jobs.append(job(env, "mfqlearning", 3))
        return jobs

    if env == "distribution":
        jobs = [job(env, "reinforce", 5)]
        jobs.append(job(env, "mfreinforce", 5, perturbation=2.0))
        jobs.extend(
            job(env, "transport", 5, perturbation=lambda_, eta=DEFAULT_TRANSPORT_ETA)
            for lambda_ in (0.1, 0.2, 0.4)
        )
        jobs.append(job(env, "adaptive_transport", 5, perturbation=ADAPTIVE_INITIAL_LAMBDA, eta=ADAPTIVE_INITIAL_ETA))
        return jobs

    if env == "advertising":
        jobs = [job(env, "reinforce", 5)]
        jobs.append(job(env, "mfreinforce", 5, perturbation=1.0))
        jobs.extend(
            job(env, "transport", 5, perturbation=lambda_, eta=DEFAULT_TRANSPORT_ETA)
            for lambda_ in (0.1, 0.2, 0.4)
        )
        jobs.append(job(env, "adaptive_transport", 5, perturbation=ADAPTIVE_INITIAL_LAMBDA, eta=ADAPTIVE_INITIAL_ETA))
        return jobs

    if env == "lq":
        jobs = [job(env, "reinforce", 20)]
        for flow in ("exact", "particle"):
            jobs.extend(
                job(env, "transport", 20, flow=flow, perturbation=lambda_, eta=DEFAULT_TRANSPORT_ETA)
                for lambda_ in TRANSPORT_LAMBDAS
            )
            jobs.append(
                job(
                    env,
                    "adaptive_transport",
                    20,
                    flow=flow,
                    perturbation=ADAPTIVE_INITIAL_LAMBDA,
                    eta=ADAPTIVE_INITIAL_ETA,
                )
            )
        return jobs

    if env == "portfolio":
        jobs = [job(env, "reinforce", 10)]
        jobs.extend(
            job(env, "transport", 10, perturbation=lambda_, eta=DEFAULT_TRANSPORT_ETA)
            for lambda_ in (0.025, 0.05, 0.1, 0.2, 0.4)
        )
        jobs.append(job(env, "adaptive_transport", 10, perturbation=ADAPTIVE_INITIAL_LAMBDA, eta=ADAPTIVE_INITIAL_ETA))
        return jobs

    if env == "kuramoto":
        jobs = [job(env, "reinforce", 20, flow="particle")]
        jobs.extend(
            job(env, "transport", 20, flow="particle", perturbation=lambda_, eta=DEFAULT_TRANSPORT_ETA)
            for lambda_ in (0.1, 0.2, 0.4)
        )
        jobs.append(
            job(
                env,
                "adaptive_transport",
                20,
                flow="particle",
                perturbation=ADAPTIVE_INITIAL_LAMBDA,
                eta=ADAPTIVE_INITIAL_ETA,
            )
        )
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


def fair_run_parameters(
    job_spec,
    adaptive_checkpoint_interval=ADAPTIVE_CHECKPOINT_INTERVAL,
    adaptive_replications=ADAPTIVE_REPLICATIONS,
):
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
    elif algorithm in {"transport", "adaptive_transport"}:
        adaptive_factor = 1.0
        if algorithm == "adaptive_transport":
            if adaptive_checkpoint_interval:
                adaptive_factor += 2.0 * adaptive_replications / adaptive_checkpoint_interval
        per_step_budget = (base_cost / horizon) / adaptive_factor
        auxiliary_gradient = TRANSPORT_AUXILIARY_GRADIENTS.get(env)
        if auxiliary_gradient is None:
            scale = per_step_budget / (ref_particles + ref_gradient)
            if algorithm == "adaptive_transport":
                auxiliary_gradient = max(1, int(scale * ref_gradient))
                parameters["n_particles"] = max(1, int(per_step_budget - auxiliary_gradient))
            else:
                parameters["n_particles"] = max(1, round(scale * ref_particles))
                auxiliary_gradient = max(1, round(scale * ref_gradient))
        else:
            auxiliary_gradient = min(auxiliary_gradient, max(1, int(per_step_budget) - 1))
            parameters["n_particles"] = max(1, int(per_step_budget - auxiliary_gradient))

        if env in DISCRETE_REFERENCE:
            parameters["n_logit_gradient"] = auxiliary_gradient
        else:
            parameters["n_law_gradient"] = auxiliary_gradient
    elif algorithm == "mfqlearning":
        parameters["n_train"] = round(base_cost * (reference["n_train"] if "n_train" in reference else 20_000))

    if job_spec["flow"] == "particle" and algorithm in {"mfreinforce", "transport", "adaptive_transport"}:
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
    if job_spec["algorithm"] in {"transport", "adaptive_transport"}:
        eta = args.eta if args.eta is not None else job_spec.get("eta")
        if eta is not None:
            command.extend(["--eta", str(eta)])

    fair_parameters = (
        fair_run_parameters(
            job_spec,
            adaptive_checkpoint_interval=args.adaptive_checkpoint_interval or ADAPTIVE_CHECKPOINT_INTERVAL,
            adaptive_replications=args.adaptive_replications or ADAPTIVE_REPLICATIONS,
        )
        if args.budget_mode == "fair"
        else {}
    )

    optional_values = {
        "--device": args.device,
        "--n-train": args.n_train if args.n_train is not None else fair_parameters.get("n_train"),
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
        "--simplex-resolution": args.simplex_resolution,
        "--q-learning-lr-power": args.q_learning_lr_power,
        "--q-learning-sampling": args.q_learning_sampling,
        "--adaptive-checkpoint-interval": args.adaptive_checkpoint_interval,
        "--adaptive-replications": args.adaptive_replications,
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
        choices=["twostate", "cybersecurity", "distribution", "advertising", "lq", "kuramoto", "portfolio", "all"],
        required=True,
    )
    parser.add_argument("--seeds", type=parse_seed_list, default=[0, 1, 2, 3, 4])
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--budget-mode", choices=["fair", "manual"], default="fair")
    parser.add_argument("--device", default=None)
    parser.add_argument("--n-train", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--eta", type=float, default=None)
    parser.add_argument("--n-particles", type=int, default=None)
    parser.add_argument("--n-logit-gradient", type=int, default=None)
    parser.add_argument("--n-law-gradient", type=int, default=None)
    parser.add_argument("--n-law-particles", type=int, default=None)
    parser.add_argument("--n-flow-particles", type=int, default=None)
    parser.add_argument("--validation-interval", type=int, default=None)
    parser.add_argument("--simplex-sigma", type=float, default=None)
    parser.add_argument("--simplex-resolution", type=int, default=None)
    parser.add_argument("--q-learning-lr-power", type=float, default=None)
    parser.add_argument("--q-learning-sampling", choices=["sweep", "iid"], default=None)
    parser.add_argument("--adaptive-checkpoint-interval", type=int, default=None)
    parser.add_argument("--adaptive-replications", type=int, default=None)
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

    envs = ["twostate", "cybersecurity", "distribution", "advertising", "lq", "kuramoto", "portfolio"]
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
