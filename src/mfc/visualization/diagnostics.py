import pandas as pd
import torch

from mfc.algorithms import (
    ContinuousTransport,
    ContinuousTransportConfig,
    DiscreteTransport,
    DiscreteTransportConfig,
)

from .io import load_env_and_policy, run_label


def vector_std_norm(values):
    if values.shape[0] < 2:
        return torch.full((), float("nan"), dtype=values.dtype)
    return values.std(dim=0, unbiased=True).norm()


def z_ratio(signal, standard_error):
    """Return a norm-to-SE ratio.

    For vector norms this is not a conventional signed z-score: under a zero
    mean error, its root-mean-square null baseline is about 1 rather than 0.
    Prefer the accompanying chi-square columns for a null-aware readout.
    """
    signal = torch.as_tensor(signal)
    standard_error = torch.as_tensor(standard_error)
    if not torch.isfinite(standard_error).item() or standard_error.item() <= 0:
        return float("nan")
    return float(signal / standard_error)


def norm_chi_square(signal, standard_error, dimension):
    """Approximate a vector norm-to-SE ratio as a chi-square statistic."""
    ratio = z_ratio(signal, standard_error)
    if dimension <= 0 or not torch.isfinite(torch.tensor(ratio)).item():
        return float("nan"), float("nan")

    statistic = dimension * ratio**2
    pvalue = torch.special.gammaincc(
        torch.tensor(0.5 * dimension, dtype=torch.float64),
        torch.tensor(0.5 * statistic, dtype=torch.float64),
    )
    return float(statistic), float(pvalue)


def reward_gradient(env, policy, lambda_):
    gradient = env.exact_gradient(policy, lambda_=lambda_)
    if type(env).__name__ == "LQ":
        return -gradient
    return gradient


def safe_scalar_ratio(numerator, denominator):
    numerator = torch.as_tensor(numerator)
    denominator = torch.as_tensor(denominator)
    if not torch.isfinite(denominator).item() or denominator.item() <= 0:
        return float("nan")
    return float(numerator / denominator)


def gradient_diagnostics(run, n_replications=20, seed=0, compare_to_unperturbed=True, n_particles=None):
    env, policy = load_env_and_policy(run)
    metadata = run["metadata"]
    algorithm_config = metadata["algorithm_config"]
    if metadata["algorithm"] != "transport":
        raise ValueError("Gradient diagnostics are defined for transport runs.")
    if not hasattr(env, "exact_gradient"):
        raise ValueError("This environment does not expose exact_gradient.")
    if isinstance(policy, torch.nn.Module):
        raise ValueError("Exact gradient diagnostics currently require direct tensor policies.")

    lambda_ = metadata["perturbation"]
    particle_count = n_particles or algorithm_config.get("n_particles")
    if metadata["env"] in {"lq", "portfolio"}:
        config = ContinuousTransportConfig(
            n_particles=particle_count,
            n_law_gradient=algorithm_config.get("n_law_gradient"),
            n_law_particles=algorithm_config.get("n_law_particles"),
            lambda_=lambda_,
            eta=algorithm_config.get("eta"),
            rho=algorithm_config.get("rho"),
            horizon=metadata["horizon"],
            flow=metadata["flow"],
            n_flow_particles=algorithm_config.get("n_flow_particles"),
            law_chart=algorithm_config.get("law_chart") or ContinuousTransportConfig.law_chart,
            min_affine_scale=algorithm_config.get("min_affine_scale"),
            baseline=algorithm_config.get("baseline", True),
            reuse_state_gradient=algorithm_config.get("reuse_state_gradient", True),
            seed=seed,
        )
        estimator = ContinuousTransport(env, policy=policy, config=config)
    else:
        config = DiscreteTransportConfig(
            n_particles=particle_count,
            n_logit_gradient=algorithm_config.get("n_logit_gradient"),
            lambda_=lambda_,
            eta=algorithm_config.get("eta"),
            horizon=metadata["horizon"],
            flow=metadata["flow"],
            n_flow_particles=algorithm_config.get("n_flow_particles"),
            simplex_sigma=algorithm_config.get("simplex_sigma", DiscreteTransportConfig.simplex_sigma),
            baseline=algorithm_config.get("baseline", True),
            reuse_state_gradient=algorithm_config.get("reuse_state_gradient", True),
            seed=seed,
        )
        estimator = DiscreteTransport(env, policy=policy, config=config)

    estimates = []
    for index in range(n_replications):
        gradient, _ = estimator.estimate_gradient(seed + index * 100_000)
        estimates.append(gradient.detach().reshape(-1).cpu())

    estimates = torch.stack(estimates)
    exact = reward_gradient(env, policy, lambda_=lambda_).detach().reshape(-1).cpu()
    error = estimates - exact
    mean_estimate = estimates.mean(dim=0)
    bias_norm = (mean_estimate - exact).norm()
    estimate_std = vector_std_norm(estimates)
    estimate_se = estimate_std / n_replications**0.5
    n_parameters = exact.numel()
    bias_chi2_stat, bias_chi2_pvalue = norm_chi_square(bias_norm, estimate_se, n_parameters)
    exact_norm = exact.norm()

    row = {
        "env": metadata["env"],
        "label": run_label(metadata),
        "flow": metadata["flow"],
        "horizon": metadata["horizon"],
        "lambda": lambda_,
        "n_parameters": n_parameters,
        "diagnostic_n_particles": estimator.n_particles,
        "n_replications": n_replications,
        "bias_norm": float(bias_norm),
        "estimate_std": float(estimate_std),
        "estimate_se": float(estimate_se),
        "estimate_se_to_exact_norm": safe_scalar_ratio(estimate_se, exact_norm),
        "estimate_snr": safe_scalar_ratio(exact_norm, estimate_se),
        "bias_z": z_ratio(bias_norm, estimate_se),
        "bias_z_null_rms": 1.0,
        "bias_chi2_stat": bias_chi2_stat,
        "bias_chi2_df": n_parameters,
        "bias_chi2_pvalue": bias_chi2_pvalue,
        "mse": float(error.square().mean()),
        "cosine_similarity": float(torch.nn.functional.cosine_similarity(mean_estimate, exact, dim=0)),
        "exact_gradient_norm": float(exact_norm),
        "estimate_gradient_norm_mean": float(estimates.norm(dim=1).mean()),
    }
    if compare_to_unperturbed:
        exact_zero = reward_gradient(env, policy, lambda_=0.0).detach().reshape(-1).cpu()
        row["perturbation_bias_norm"] = float((exact - exact_zero).norm())

    return pd.DataFrame([row])


def transport_correction_table(run, n_replications=20, n_particles=None, seed=0):
    env, policy = load_env_and_policy(run)
    metadata = run["metadata"]
    algorithm_config = metadata["algorithm_config"]
    if metadata["algorithm"] != "transport":
        raise ValueError("Correction diagnostics are defined for transport runs.")

    lambda_ = metadata["perturbation"]
    eta = algorithm_config.get("eta") or lambda_
    particle_count = n_particles or algorithm_config.get("n_particles") or env.config.n_particles

    if metadata["env"] in {"lq", "portfolio"}:
        config = ContinuousTransportConfig(
            n_particles=particle_count,
            n_law_gradient=algorithm_config.get("n_law_gradient"),
            n_law_particles=algorithm_config.get("n_law_particles"),
            lambda_=lambda_,
            eta=eta,
            rho=algorithm_config.get("rho"),
            horizon=metadata["horizon"],
            flow=metadata["flow"],
            n_flow_particles=algorithm_config.get("n_flow_particles"),
            law_chart=algorithm_config.get("law_chart") or ContinuousTransportConfig.law_chart,
            min_affine_scale=algorithm_config.get("min_affine_scale"),
            reuse_state_gradient=algorithm_config.get("reuse_state_gradient", True),
            seed=seed,
            baseline=False,
        )
        estimator = ContinuousTransport(env, policy=policy, config=config)
    else:
        config = DiscreteTransportConfig(
            n_particles=particle_count,
            n_logit_gradient=algorithm_config.get("n_logit_gradient"),
            lambda_=lambda_,
            eta=eta,
            horizon=metadata["horizon"],
            flow=metadata["flow"],
            n_flow_particles=algorithm_config.get("n_flow_particles"),
            simplex_sigma=algorithm_config.get("simplex_sigma", DiscreteTransportConfig.simplex_sigma),
            reuse_state_gradient=algorithm_config.get("reuse_state_gradient", True),
            seed=seed,
            baseline=False,
        )
        estimator = DiscreteTransport(env, policy=policy, config=config)

    rows = []
    full_gradients = []
    policy_gradients = []
    corrections = []
    for replication in range(n_replications):
        base_seed = seed + replication * 100_000

        if isinstance(estimator, ContinuousTransport):
            moments = estimator.mean_field_moment_flow(seed=base_seed + 20_000)
            sensitivities = estimator.estimate_moment_sensitivities(moments, base_seed + 10_000)
            zeros = [torch.zeros_like(sensitivity) for sensitivity in sensitivities]
            full_gradient, _ = estimator.batched_trajectory_gradient(moments, sensitivities, base_seed)
            policy_gradient, _ = estimator.batched_trajectory_gradient(moments, zeros, base_seed)
        else:
            laws, _ = estimator.mean_field_law_flow(seed=base_seed + 20_000)
            sensitivities = estimator.estimate_state_sensitivities(laws, base_seed + 10_000)
            zeros = [torch.zeros_like(sensitivity) for sensitivity in sensitivities]
            full_gradient, _ = estimator.batched_trajectory_gradient(laws, sensitivities, base_seed)
            policy_gradient, _ = estimator.batched_trajectory_gradient(laws, zeros, base_seed)

        full_gradient = full_gradient.detach().cpu()
        policy_gradient = policy_gradient.detach().cpu()
        correction = full_gradient - policy_gradient
        full_gradients.append(full_gradient)
        policy_gradients.append(policy_gradient)
        corrections.append(correction)
        rows.append(
            {
                "env": metadata["env"],
                "label": run_label(metadata),
                "flow": metadata["flow"],
                "horizon": metadata["horizon"],
                "seed": metadata["seed"],
                "lambda": lambda_,
                "replication": replication,
                "full_gradient_norm": float(full_gradient.norm()),
                "policy_only_gradient_norm": float(policy_gradient.norm()),
                "correction_norm": float(correction.norm()),
                "correction_fraction": float(correction.norm() / full_gradient.norm().clamp_min(1e-12)),
                "full_policy_cosine": float(torch.nn.functional.cosine_similarity(full_gradient, policy_gradient, dim=0)),
            }
        )

    table = pd.DataFrame(rows)
    if table.empty:
        return table

    full_gradients = torch.stack(full_gradients)
    policy_gradients = torch.stack(policy_gradients)
    corrections = torch.stack(corrections)
    correction_mean = corrections.mean(dim=0)
    correction_std = vector_std_norm(corrections)
    correction_se = correction_std / n_replications**0.5
    full_mean = full_gradients.mean(dim=0)
    policy_mean = policy_gradients.mean(dim=0)
    n_parameters = correction_mean.numel()
    correction_chi2_stat, correction_chi2_pvalue = norm_chi_square(
        correction_mean.norm(),
        correction_se,
        n_parameters,
    )

    table["n_parameters"] = n_parameters
    table["full_gradient_mean_norm"] = float(full_mean.norm())
    table["policy_only_gradient_mean_norm"] = float(policy_mean.norm())
    table["correction_mean_norm"] = float(correction_mean.norm())
    table["correction_std"] = float(correction_std)
    table["correction_se"] = float(correction_se)
    table["correction_z"] = z_ratio(correction_mean.norm(), correction_se)
    table["correction_z_null_rms"] = 1.0
    table["correction_chi2_stat"] = correction_chi2_stat
    table["correction_chi2_df"] = n_parameters
    table["correction_chi2_pvalue"] = correction_chi2_pvalue
    table["correction_mean_fraction"] = float(correction_mean.norm() / full_mean.norm().clamp_min(1e-12))
    table["full_policy_mean_cosine"] = float(torch.nn.functional.cosine_similarity(full_mean, policy_mean, dim=0))
    return table
