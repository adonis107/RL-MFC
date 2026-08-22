import matplotlib.pyplot as plt
import pandas as pd
import torch

from .constants import STATE_LABELS
from .flows import final_policy_probabilities, learned_flow
from .io import load_env_and_policy, validation_dataframe


def use_symlog_validation_axis(ax, values, threshold_ratio=1_000.0):
    finite = pd.Series(values, dtype="float64").dropna()
    finite = finite[(finite != float("inf")) & (finite != float("-inf"))]
    magnitudes = finite.abs()
    magnitudes = magnitudes[magnitudes > 0]
    if magnitudes.empty:
        return False

    if magnitudes.max() / magnitudes.min() < threshold_ratio:
        return False

    ax.set_yscale("symlog", linthresh=max(1.0, float(magnitudes.min())))
    return True


def plot_validation_rewards(runs, env=None, horizon=None, flow=None, ax=None, save_path=None):
    df = validation_dataframe(runs)
    if env is not None:
        df = df[df["env"] == env]
    if horizon is not None:
        df = df[df["horizon"] == horizon]
    if flow is not None:
        df = df[df["flow"] == flow]
    if df.empty:
        raise ValueError("No validation data matched the requested filters.")

    df = df.copy()
    label_parts = [df["label"]]
    if flow is None and df["flow"].nunique(dropna=False) > 1:
        label_parts.append(df["flow"].map(lambda value: f"flow={value}"))
    if horizon is None and df["horizon"].nunique(dropna=False) > 1:
        label_parts.append(df["horizon"].map(lambda value: f"T={value}"))
    df["plot_label"] = label_parts[0]
    for part in label_parts[1:]:
        df["plot_label"] = df["plot_label"] + ", " + part

    summary = (
        df.groupby(["plot_label", "step"], as_index=False)["validation_reward"]
        .agg(["mean", "std"])
        .reset_index()
        .fillna({"std": 0.0})
    )

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4.5))

    for label, group in summary.groupby("plot_label", sort=False):
        group = group.sort_values("step")
        x = group["step"].to_numpy()
        mean = group["mean"].to_numpy()
        std = group["std"].to_numpy()
        ax.plot(x, mean, label=label)
        ax.fill_between(x, mean - std, mean + std, alpha=0.18)

    ax.set_xlabel("training step")
    ax.set_ylabel("validation reward")
    if env == "lq":
        use_symlog_validation_axis(ax, df["validation_reward"])
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    if save_path is not None:
        ax.figure.savefig(save_path, bbox_inches="tight", dpi=180)
    return ax, summary


def plot_distribution_comparison(run, ax=None, save_path=None):
    env, _ = load_env_and_policy(run)
    flow = learned_flow(run)
    metadata = run["metadata"]

    if metadata["env"] != "distribution":
        raise ValueError("plot_distribution_comparison is for the distribution planning benchmark.")

    labels = STATE_LABELS["distribution"]
    data = pd.DataFrame(
        {
            "initial": env.initial_distribution.cpu().numpy(),
            "target": env.target_distribution.cpu().numpy(),
            "learned": flow[-1].numpy(),
        },
        index=labels,
    )

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4.5))
    data.plot(kind="bar", ax=ax)
    ax.set_xlabel("state")
    ax.set_ylabel("probability")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    if save_path is not None:
        ax.figure.savefig(save_path, bbox_inches="tight", dpi=180)
    return ax, data


def plot_state_flow(run, ax=None, save_path=None):
    metadata = run["metadata"]
    flow = learned_flow(run)

    if metadata["env"] in {"lq", "portfolio"}:
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 4.5))
        steps = range(flow.shape[0])
        ax.plot(steps, flow[:, 0], label="mean")
        ax.plot(steps, flow[:, 1], label="variance")
        ax.set_xlabel("time")
        ax.set_ylabel("moment")
        ax.legend(frameon=False)
        ax.grid(alpha=0.25)
        if save_path is not None:
            ax.figure.savefig(save_path, bbox_inches="tight", dpi=180)
        return ax, pd.DataFrame({"time": list(steps), "mean": flow[:, 0].numpy(), "variance": flow[:, 1].numpy()})

    labels = STATE_LABELS.get(metadata["env"], [str(i) for i in range(flow.shape[1])])
    df = pd.DataFrame(flow.numpy(), columns=labels)
    df["time"] = range(len(df))

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4.5))
    for label in labels:
        ax.plot(df["time"], df[label], label=label)
    ax.set_xlabel("time")
    ax.set_ylabel("state probability")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    if save_path is not None:
        ax.figure.savefig(save_path, bbox_inches="tight", dpi=180)
    return ax, df


def plot_advertising_diagnostics(run, ax=None, save_path=None):
    metadata = run["metadata"]
    if metadata["env"] != "advertising":
        raise ValueError("plot_advertising_diagnostics is for the advertising benchmark.")

    env, policy = load_env_and_policy(run)
    flow = learned_flow(run).to(env.device)
    customer = flow[:, env.CUSTOMER].cpu()
    ad_probability = []
    with torch.no_grad():
        for t in range(metadata["horizon"]):
            probs = final_policy_probabilities(env, policy, law=flow[t], t=t)
            ad_probability.append(float(probs[:, env.AD].mean()))

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(range(len(customer)), customer, label="customer share")
    ax.plot(range(len(ad_probability)), ad_probability, label="mean ad probability")
    ax.set_xlabel("time")
    ax.set_ylabel("value")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    if save_path is not None:
        ax.figure.savefig(save_path, bbox_inches="tight", dpi=180)
    return ax, pd.DataFrame({"time": range(len(customer)), "customer_share": customer.numpy()})


def plot_flow_comparison(runs, env, horizon=None, ax=None, save_path=None):
    df = validation_dataframe(runs)
    df = df[(df["env"] == env) & (df["algorithm"] == "transport")]
    if horizon is not None:
        df = df[df["horizon"] == horizon]
    if df.empty:
        raise ValueError("No transport validation data matched the requested filters.")

    grouped = (
        df.groupby(["perturbation", "flow", "step"], as_index=False)["validation_reward"]
        .agg(["mean", "std"])
        .reset_index()
        .fillna({"std": 0.0})
    )

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4.5))

    for (lambda_, flow), group in grouped.groupby(["perturbation", "flow"], sort=False):
        group = group.sort_values("step")
        label = f"lambda={lambda_:g}, {flow}"
        ax.plot(group["step"], group["mean"], label=label)

    ax.set_xlabel("training step")
    ax.set_ylabel("validation reward")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    if save_path is not None:
        ax.figure.savefig(save_path, bbox_inches="tight", dpi=180)
    return ax, grouped
