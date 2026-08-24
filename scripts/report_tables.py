"""Emit the LaTeX tables of the numerical-experiments chapter from saved results.

Every table in Chapter~\\ref{chapter:experiments} and in the experimental appendix is
produced here from the CSV summaries written by ``scripts/plot.py``, so that the
reported numbers can be regenerated from the saved runs rather than transcribed.

    uv run python scripts/report_tables.py --figures-root results/figures --output-root files/report/tables
"""

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

# Kuramoto is excluded: it is discussed only as a possible future benchmark.
ENVIRONMENTS = ["twostate", "cybersecurity", "distribution", "advertising", "lq", "portfolio"]

DISPLAY_NAME = {
    "twostate": "Two-state",
    "cybersecurity": "Cybersecurity",
    "distribution": "Distribution planning",
    "advertising": "Targeted advertising",
    "lq": "Linear--quadratic",
    "portfolio": "Portfolio",
}

# Population-flow mode used in the headline comparison of each benchmark.
MAIN_FLOW = {
    "twostate": "exact",
    "cybersecurity": "exact",
    "distribution": "exact",
    "advertising": "exact",
    "lq": "exact",
    "portfolio": "exact",
}

MAIN_HORIZON = {
    "twostate": 5,
    "cybersecurity": 3,
    "distribution": 5,
    "advertising": 5,
    "lq": 20,
    "portfolio": 10,
}

# Reference optima, in the sign convention of the reported validation objective.
# The two-state values are the exact recursion evaluated at the closed-form optimal
# policy from the fixed validation law; the others come from ``J0_star``.
REFERENCE_OPTIMUM = {
    ("twostate", 2): -3.360,
    ("twostate", 5): -2.640,
}



# Main and auxiliary trajectory counts of the fixed-scale transport runs, read from the
# saved algorithm configuration of each benchmark at its headline configuration.
TRANSPORT_ALLOCATION = {
    "twostate": (248, 12),
    "cybersecurity": (184, 20),
    "distribution": (549, 11),
    "advertising": (248, 12),
    "lq": (201, 20),
    "portfolio": (311, 200),
}


def load(figures_root, env, name):
    path = Path(figures_root) / env / f"{name}.csv"
    return pd.read_csv(path) if path.exists() else None


def reference_optimum(figures_root, env, horizon):
    if (env, horizon) in REFERENCE_OPTIMUM:
        return REFERENCE_OPTIMUM[(env, horizon)]
    table = load(figures_root, env, "objectives")
    if table is None or "J0_star" not in table.columns:
        return None
    values = table.loc[table["horizon"] == horizon, "J0_star"].dropna()
    if values.empty:
        return None
    convention = table["objective_convention"].dropna().iloc[0]
    return -float(values.iloc[0]) if convention == "cost" else float(values.iloc[0])


def method_of(label):
    if label.startswith("REINFORCE"):
        return "reinforce"
    if label.startswith("MF-REINFORCE"):
        return "mfreinforce"
    if label.startswith("Adaptive"):
        return "adaptive"
    return "transport"


def grouped_objectives(figures_root, env):
    table = load(figures_root, env, "objectives")
    grouped = (
        table.groupby(["label", "flow", "horizon"])["validation_reward"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    grouped["method"] = grouped["label"].map(method_of)
    return grouped


def selection(grouped, env, horizon=None, flow=None):
    """Best entry per method at the headline horizon and flow of a benchmark."""
    horizon = MAIN_HORIZON[env] if horizon is None else horizon
    flow = MAIN_FLOW[env] if flow is None else flow
    subset = grouped[grouped["horizon"] == horizon]
    # REINFORCE never uses a perturbed population flow, so it is always stored as exact.
    subset = subset[(subset["flow"] == flow) | (subset["method"] == "reinforce")]
    best = {}
    for method, rows in subset.groupby("method"):
        best[method] = rows.loc[rows["mean"].idxmax()]
    return best


def number(value, digits=4):
    return "---" if value is None or pd.isna(value) else f"{value:.{digits}f}"


def with_error(row, digits=4):
    if row is None:
        return "---"
    return f"${number(row['mean'], digits)}\\pm{number(row['std'], digits)}$"


def transport_scales(label):
    """``Transport lambda=0.1, eta=0.85`` -> ``$\\lambda=0.1$, $\\eta=0.85$``."""
    body = label.split("lambda=", 1)[1]
    lambda_text, eta_text = (part.strip() for part in body.split(", eta="))
    return f"$\\lambda={lambda_text}$, $\\eta={eta_text}$"


def pretty_label(label):
    """Rewrite a run label with the mathematical notation used in the chapter."""
    if label.startswith("Transport "):
        return f"Transport {transport_scales(label)}"
    if label.startswith("Adaptive"):
        body = label.split("lambda0=", 1)[1]
        lambda_text, eta_text = (part.strip() for part in body.split(", eta0="))
        return f"Adaptive $\\lambda_0={lambda_text}$, $\\eta_0={eta_text}$"
    if label.startswith("MF-REINFORCE"):
        return f"MF-REINFORCE $\\varepsilon={label.split('eps=', 1)[1].strip()}$"
    return label


def table_environment(body, caption, label, alignment, size=None):
    return "\n".join(
        [
            "\\begin{table}[H]",
            "\\centering",
            *([size] if size else []),
            f"\\begin{{tabular}}{{{alignment}}}",
            "\\toprule",
            body,
            "\\bottomrule",
            "\\end{tabular}",
            f"\\caption{{{caption}}}",
            f"\\label{{{label}}}",
            "\\end{table}",
            "",
        ]
    )


def objective_summary(figures_root):
    """Headline table: one row per benchmark, one column per estimator family."""
    short_name = {
        "distribution": "Distribution",
        "advertising": "Advertising",
    }
    lines = [
        "Benchmark & Optimum & REINFORCE & MF-REINFORCE & Transport & $(\\lambda,\\eta)$ & Adaptive \\\\",
        "\\midrule",
    ]
    for env in ENVIRONMENTS:
        grouped = grouped_objectives(figures_root, env)
        horizon = MAIN_HORIZON[env]
        best = selection(grouped, env)
        optimum = reference_optimum(figures_root, env, horizon)
        digits = 4 if env != "portfolio" else 3
        transport = best.get("transport")
        if transport is None:
            scales = "---"
        else:
            body = transport["label"].split("lambda=", 1)[1]
            lambda_text, eta_text = (part.strip() for part in body.split(", eta="))
            scales = f"$({lambda_text},{eta_text})$"
        lines.append(
            " & ".join(
                [
                    f"{short_name.get(env, DISPLAY_NAME[env])} ($T={horizon}$)",
                    "---" if optimum is None else f"${number(optimum, digits)}$",
                    with_error(best.get("reinforce"), digits),
                    with_error(best.get("mfreinforce"), digits),
                    with_error(transport, digits),
                    scales,
                    with_error(best.get("adaptive"), digits),
                ]
            )
            + " \\\\"
        )
    caption = (
        "Final validation objective on every benchmark, as mean and standard deviation over five seeds, "
        "at the headline horizon and population-flow mode of each subsection. Higher is better throughout. "
        "The transport column reports the best entry of the fixed-scale grid and the scales $(\\lambda,\\eta)$ that attain it; "
        "MF-REINFORCE is not defined on the continuous-state benchmarks. "
        "The optimum column gives the exact recursion evaluated at the closed-form optimal policy where one is available."
    )
    return table_environment(
        "\n".join(lines),
        caption,
        "tab:objective-summary",
        "lrrrrlr",
        size="\\footnotesize\n\\setlength{\\tabcolsep}{2.6pt}",
    )


def budget_runtime(figures_root):
    """Simulator budgets are matched by construction; wall-clock cost is not."""
    lines = [
        "Benchmark & Estimator & Simulator budget & Wall clock (s) & Ratio to REINFORCE \\\\",
        "\\midrule",
    ]
    for env in ENVIRONMENTS:
        runtime = load(figures_root, env, "runtime")
        horizon, flow = MAIN_HORIZON[env], MAIN_FLOW[env]
        subset = runtime[runtime["horizon"] == horizon]
        subset = subset[(subset["flow"] == flow) | (subset["algorithm"] == "reinforce")]
        reference = subset[subset["algorithm"] == "reinforce"]
        reference_seconds = float(reference["elapsed_seconds_mean"].iloc[0]) if not reference.empty else None
        rows = []
        for algorithm, name in [
            ("reinforce", "REINFORCE"),
            ("mfreinforce", "MF-REINFORCE"),
            ("transport", "Transport"),
            ("adaptive_transport", "Adaptive"),
        ]:
            entries = subset[subset["algorithm"] == algorithm]
            if entries.empty:
                continue
            entry = entries.iloc[entries["elapsed_seconds_mean"].to_numpy().argmax()]
            ratio = (
                "---"
                if reference_seconds is None
                else f"${float(entry['elapsed_seconds_mean']) / reference_seconds:.1f}\\times$"
            )
            rows.append(
                " & ".join(
                    [
                        DISPLAY_NAME[env] if not rows else "",
                        name,
                        f"${float(entry['simulator_budget_mean']):,.0f}$".replace(",", "\\,"),
                        f"${float(entry['elapsed_seconds_mean']):,.0f}$".replace(",", "\\,"),
                        ratio,
                    ]
                )
                + " \\\\"
            )
        lines.extend(rows)
        if env != ENVIRONMENTS[-1]:
            lines.append("\\midrule")
    caption = (
        "Simulator budget and wall-clock cost per run at the headline configuration, averaged over five seeds. "
        "The budget is matched by construction on every benchmark in the table. "
        "For transport and adaptive transport the slowest entry of the grid is reported. "
        "Matching the simulator budget does not match the wall-clock cost: the transport update carries an "
        "auxiliary sensitivity estimate that is cheap in simulated transitions and expensive in arithmetic."
    )
    return table_environment("\n".join(lines), caption, "tab:budget-runtime", "llrrr")


def twostate_horizon(figures_root):
    """The horizon comparison: objective gap and policy-recovery error at T=2 and T=5."""
    objectives = load(figures_root, "twostate", "objectives")
    grouped = grouped_objectives(figures_root, "twostate")
    errors = load(figures_root, "twostate", "policy_error")

    lines = [
        "& & \\multicolumn{2}{c}{$J(\\theta^\\star)-J(\\widehat\\theta)$} "
        "& \\multicolumn{2}{c}{$\\tfrac12(e_0+e_1)$} \\\\",
        "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}",
        "Estimator & Scales & $T=2$ & $T=5$ & $T=2$ & $T=5$ \\\\",
        "\\midrule",
    ]
    for method, name in [
        ("reinforce", "REINFORCE"),
        ("mfreinforce", "MF-REINFORCE"),
        ("transport", "Transport"),
        ("adaptive", "Adaptive transport"),
    ]:
        gaps, policy, scales = [], [], []
        for horizon in (2, 5):
            best = selection(grouped, "twostate", horizon=horizon, flow="exact")[method]
            optimum = REFERENCE_OPTIMUM[("twostate", horizon)]
            # Per-seed gap: the optimum is a deterministic constant, so all the
            # dispersion below comes from the learned policy.
            seeds = objectives[
                (objectives["label"] == best["label"])
                & (objectives["flow"] == "exact")
                & (objectives["horizon"] == horizon)
            ]["validation_reward"]
            gap = optimum - seeds
            gaps.append((gap.mean(), gap.std()))
            seed_errors = errors[
                (errors["label"] == best["label"])
                & (errors["flow"] == "exact")
                & (errors["horizon"] == horizon)
            ]["mean_abs_policy_error"]
            policy.append((seed_errors.mean(), seed_errors.std()))
            if method == "transport":
                scales.append(transport_scales(best["label"]).replace("$", "").replace("\\lambda", "$\\lambda"))
        if method == "transport":
            best2 = selection(grouped, "twostate", horizon=2, flow="exact")[method]["label"]
            best5 = selection(grouped, "twostate", horizon=5, flow="exact")[method]["label"]
            def pair(label):
                body = label.split("lambda=", 1)[1]
                lambda_text, eta_text = (part.strip() for part in body.split(", eta="))
                return f"$({lambda_text},{eta_text})$"

            scale_cell = "{\\footnotesize " + pair(best2) + " / " + pair(best5) + "}"
        elif method == "adaptive":
            scale_cell = "{\\footnotesize $(0.2,0.8)$}"
        elif method == "mfreinforce":
            scale_cell = "{\\footnotesize $\\varepsilon=0.2$}"
        else:
            scale_cell = "---"
        lines.append(
            " & ".join(
                [
                    name,
                    scale_cell,
                    f"${gaps[0][0]:.3f}\\pm{gaps[0][1]:.3f}$",
                    f"${gaps[1][0]:.3f}\\pm{gaps[1][1]:.3f}$",
                    f"${policy[0][0]:.4f}\\pm{policy[0][1]:.4f}$",
                    f"${policy[1][0]:.4f}\\pm{policy[1][1]:.4f}$",
                ]
            )
            + " \\\\"
        )
    caption = (
        "Two-state benchmark with exact population flow, at matched simulator budgets. "
        "Both columns are evaluated at the frozen learned parameter $\\widehat\\theta$ and reported as mean and "
        "standard deviation over the five seeds. "
        "The objective column is the exact unperturbed objective of the deterministic recursion "
        "\\eqref{eq:twostate-population-recursion} started from the fixed validation law "
        "$\\mu_0^{\\mathrm{val}}=(0.2,0.8)$, evaluated at the closed-form optimum $\\theta^\\star$ of "
        "\\eqref{eq:twostate-objective} and at $\\widehat\\theta$, and differenced: "
        "$J(\\theta^\\star;\\mu_0^{\\mathrm{val}})=-3.360$ at $T=2$ and $-2.640$ at $T=5$. "
        "Neither term carries Monte Carlo error, so the dispersion is entirely across seeds of $\\widehat\\theta$. "
        "The policy column is the average of the two state-wise recovery errors $e_0$ and $e_1$ defined above. For transport the best entry of the $(\\lambda,\\eta)$ grid is reported at "
        "each horizon, and the two winning pairs $(\\lambda,\\eta)$ are listed in the scales column as $T=2$ / $T=5$; that column carries $\\varepsilon$ for MF-REINFORCE and $(\\lambda_0,\\eta_0)$ for the adaptive controller. "
        "MF-REINFORCE is the more accurate estimator at $T=2$ and loses to transport at $T=5$, on both measures."
    )
    return table_environment("\n".join(lines), caption, "tab:twostate-horizon", "llrrrr", size="\\small")


def twostate_grid(figures_root, horizon=5, flow="exact"):
    """The (lambda, eta) calibration grid."""
    table = load(figures_root, "twostate", "objectives")
    table = table[
        (table["horizon"] == horizon)
        & (table["flow"] == flow)
        & (table["label"].str.startswith("Transport"))
    ].copy()
    table["lambda"] = table["label"].str.extract(r"lambda=([\d.]+)").astype(float)
    grid = table.pivot_table(index="lambda", columns="eta", values="validation_reward", aggfunc="mean")

    etas = list(grid.columns)
    best = grid.to_numpy().max()
    lines = [
        "& \\multicolumn{" + str(len(etas)) + "}{c}{Auxiliary scale $\\eta$} \\\\",
        f"\\cmidrule(lr){{2-{len(etas) + 1}}}",
        "$\\lambda$ & " + " & ".join(f"${eta:g}$" for eta in etas) + " \\\\",
        "\\midrule",
    ]
    for lambda_value, row in grid.iterrows():
        cells = []
        for eta in etas:
            cells.append(f"$\\mathbf{{{row[eta]:.3f}}}$" if row[eta] == best else f"${row[eta]:.3f}$")
        lines.append(f"${lambda_value:g}$ & " + " & ".join(cells) + " \\\\")
    caption = (
        f"Two-state calibration grid at $T={horizon}$ with exact population flow: final validation objective, "
        "averaged over five seeds, for every pair of perturbation scales. The optimum is interior in $\\lambda$, "
        "as the bias-variance trade-off of the main perturbation predicts. In $\\eta$ the objective increases "
        "over the whole tested range whenever $\\lambda\\leq0.1$ and peaks at $\\eta=0.85$ for $\\lambda\\in\\{0.2,0.4\\}$, "
        "so the auxiliary estimator wants a radius near the top of the grid: at the auxiliary sample sizes used, "
        "the $(n\\eta^2)^{-1}$ variance term of Proposition~\\ref{prop:auxiliary-sensitivity-error} dominates its "
        "$O(\\eta)$ bias. The ordering reverses only at $\\lambda=0.8$, where the run is already dominated by "
        "perturbation bias. This grid fixes $\\eta=0.85$ for the remaining benchmarks."
    )
    return table_environment("\n".join(lines), caption, "tab:twostate-grid", "l" + "r" * len(etas))


def decomposition(figures_root, env):
    """Optimization error on the perturbed problem against the true optimality gap.

    Available on the benchmarks that expose both the perturbed objective and its
    optimizer in closed form, so that the residual error can be split into the part
    the estimator could still remove and the part the perturbation itself carries.
    """
    table = load(figures_root, env, "objectives")
    table = table[table["flow"] == MAIN_FLOW[env]]
    cost = table["objective_convention"].dropna().iloc[0] == "cost"
    # Sign so that both columns are non-negative distances, whatever the convention.
    sign = 1.0 if cost else -1.0

    transport = table[table["label"].str.startswith("Transport")].copy()
    transport["lambda"] = transport["label"].str.extract(r"lambda=([\d.]+)").astype(float)
    transport["optimization_error"] = sign * (transport["Jlambda"] - transport["Jlambda_star"])
    transport["true_gap"] = sign * (transport["J0"] - transport["J0_star"])
    transport["perturbation_bias"] = sign * (transport["Jlambda_star"] - transport["J0_star"])

    digits = 3 if env == "lq" else 2
    lines = [
        "$\\lambda$ & $J^\\lambda(\\theta_\\lambda^\\star)$ & Perturbation bias "
        "& Optimization error & Optimality gap \\\\",
        "& & $|J^\\lambda(\\theta_\\lambda^\\star)-J(\\theta^\\star)|$ "
        "& $|J^\\lambda(\\widehat\\theta_\\lambda)-J^\\lambda(\\theta_\\lambda^\\star)|$ "
        "& $|J(\\widehat\\theta_\\lambda)-J(\\theta^\\star)|$ \\\\",
        "\\midrule",
    ]
    for lambda_value, rows in transport.groupby("lambda"):
        lines.append(
            " & ".join(
                [
                    f"${lambda_value:g}$",
                    f"${rows['Jlambda_star'].iloc[0]:.{digits}f}$",
                    f"${rows['perturbation_bias'].iloc[0]:.{digits}f}$",
                    f"${rows['optimization_error'].mean():.4f}\\pm{rows['optimization_error'].std():.4f}$",
                    f"${rows['true_gap'].mean():.4f}\\pm{rows['true_gap'].std():.4f}$",
                ]
            )
            + " \\\\"
        )

    lines.append("\\midrule")
    for prefix, name in [("Adaptive", "Adaptive"), ("REINFORCE", "REINFORCE")]:
        rows = table[table["label"].str.startswith(prefix)]
        if rows.empty:
            continue
        gap = sign * (rows["J0"] - rows["J0_star"])
        lines.append(
            f"{name} & --- & --- & --- & ${gap.mean():.4f}\\pm{gap.std():.4f}$ \\\\"
        )

    convention = "cost convention, lower is better" if cost else "reward convention, higher is better"
    optimum = transport["J0_star"].iloc[0]
    caption = (
        f"{DISPLAY_NAME[env]} benchmark at $T={MAIN_HORIZON[env]}$ with {MAIN_FLOW[env]} population flow, "
        f"{convention}, with $J(\\theta^\\star)={optimum:.3f}$. The residual error of a fixed-scale transport run "
        "splits into two parts. The perturbation bias is a property of the problem the estimator targets, not of "
        "the run: it is the distance between the optimal value of the perturbed objective and the true optimum, "
        "and it carries no seed dispersion. The optimization error is how far the learned parameter "
        "$\\widehat\\theta_\\lambda$ is from the optimizer of that perturbed objective, and it is the only part a "
        "better estimator could remove. The last column is the quantity that matters, and it is not the sum of the "
        "other two, since a policy optimized for $J^\\lambda$ need not be displaced from $\\theta^\\star$ by the "
        "full bias of the objective. All entries are means and standard deviations over five seeds, computed from "
        "closed-form expressions with no Monte Carlo error, so the dispersion is entirely across seeds. The adaptive "
        "controller varies $\\lambda$ during training and REINFORCE targets no perturbed problem, so only their "
        "optimality gaps are defined."
    )
    return table_environment("\n".join(lines), caption, f"tab:{env}-decomposition", "lrrrr", size="\\small")


def adaptive_scales(figures_root):
    """Terminal perturbation scales selected by the adaptive controller."""
    lines = [
        "Benchmark & Flow & $\\lambda$ after training & $\\eta$ after training & Best fixed $\\lambda$ & Resolved bias \\\\",
        "\\midrule",
    ]
    for env in ENVIRONMENTS:
        schedule = load(figures_root, env, "adaptive_schedule")
        if schedule is None or schedule.empty:
            continue
        grouped = grouped_objectives(figures_root, env)
        horizon, flow = MAIN_HORIZON[env], MAIN_FLOW[env]
        subset = schedule[(schedule["horizon"] == horizon) & (schedule["flow"] == flow)]
        if subset.empty:
            continue
        final = subset.sort_values("step").groupby("seed").tail(1)
        best_transport = selection(grouped, env).get("transport")
        best_lambda = (
            transport_scales(best_transport["label"]).split(",")[0].replace("$\\lambda=", "$").strip()
            if best_transport is not None
            else "---"
        )
        lines.append(
            " & ".join(
                [
                    DISPLAY_NAME[env],
                    flow,
                    f"${final['lambda'].mean():.3f}\\pm{final['lambda'].std():.3f}$",
                    f"${final['eta'].mean():.3f}\\pm{final['eta'].std():.3f}$",
                    best_lambda,
                    f"${100 * subset['lambda_resolved'].mean():.0f}\\%$",
                ]
            )
            + " \\\\"
        )
    caption = (
        "Perturbation scales reached by the adaptive controller of Algorithm~\\ref{alg:adaptive-transport}, "
        "started from $\\lambda_0=0.2$, $\\eta_0=0.8$ on every benchmark, as mean and standard deviation over "
        "five seeds. The last column gives the fraction of checkpoints at which the debiased bias estimate was "
        "resolved and the main scale was therefore allowed to move. The controller contracts both scales on every "
        "benchmark. On the continuous-state problems the $\\lambda$ it settles on is comparable to the best entry "
        "of the hand-tuned grid; on the finite-state ones it contracts below the smallest calibrated scale, which "
        "is where it underperforms."
    )
    return table_environment("\n".join(lines), caption, "tab:adaptive-scales", "llrrrr")


def benchmark_summary(figures_root, env):
    """Per-algorithm results at the headline configuration of one benchmark."""
    objectives = load(figures_root, env, "objectives")
    horizon, flow = MAIN_HORIZON[env], MAIN_FLOW[env]
    subset = objectives[
        (objectives["horizon"] == horizon)
        & ((objectives["flow"] == flow) | (objectives["label"] == "REINFORCE"))
    ]
    grouped = subset.groupby("label")["validation_reward"].agg(["mean", "std"]).reset_index()
    grouped["method"] = grouped["label"].map(method_of)
    order = {"reinforce": 0, "mfreinforce": 1, "transport": 2, "adaptive": 3}
    grouped = grouped.sort_values(["method", "label"], key=lambda column: column.map(order) if column.name == "method" else column)

    best = grouped["mean"].max()
    runtime = load(figures_root, env, "runtime")
    lines = ["Estimator & Final objective & Wall clock (s) \\\\", "\\midrule"]
    for _, row in grouped.iterrows():
        entry = runtime[
            (runtime["horizon"] == horizon)
            & ((runtime["flow"] == flow) | (runtime["algorithm"] == "reinforce"))
        ]
        algorithm = {"reinforce": "reinforce", "mfreinforce": "mfreinforce", "adaptive": "adaptive_transport"}.get(
            row["method"], "transport"
        )
        entry = entry[entry["algorithm"] == algorithm]
        if row["method"] == "transport" and not entry.empty:
            lambda_text = row["label"].split("lambda=", 1)[1].split(",")[0]
            entry = entry[entry["perturbation"].astype(str).str.startswith(lambda_text)]
        seconds = "---" if entry.empty else f"${float(entry['elapsed_seconds_mean'].iloc[0]):,.0f}$".replace(",", "\\,")
        body = f"{row['mean']:.4f}\\pm{row['std']:.4f}"
        value = f"$\\mathbf{{{body}}}$" if row["mean"] == best else f"${body}$"
        lines.append(" & ".join([pretty_label(row["label"]), value, seconds]) + " \\\\")
    caption = (
        f"{DISPLAY_NAME[env]} benchmark at $T={horizon}$ with {flow} population flow: final validation objective "
        "as mean and standard deviation over five seeds, and mean wall-clock cost per run, at a matched simulator "
        "budget. Higher is better."
    )
    return table_environment("\n".join(lines), caption, f"tab:summary-{env}", "lrr")


def auxiliary_budget(figures_root):
    """How the matched simulator budget is split between main and auxiliary trajectories."""
    # Free coordinates of the population argument: N-1 on a finite simplex, and the
    # dimension of the moment chart on the continuous-state benchmarks.
    free_coordinates = {
        "twostate": 1,
        "cybersecurity": 3,
        "distribution": 9,
        "advertising": 1,
        "lq": 1,
        "portfolio": 1,
    }
    argument = {
        "twostate": "$\\Delta_2$",
        "cybersecurity": "$\\Delta_4$",
        "distribution": "$\\Delta_{10}$",
        "advertising": "$\\Delta_2$",
        "lq": "mean",
        "portfolio": "mean",
    }
    lines = [
        "Benchmark & Argument & $d$ & $B_{\\mathrm{tr}}$ & $n_{\\mathrm{tr}}$ & $n_{\\mathrm{tr}}/d$ & Budget \\\\",
        "\\midrule",
    ]
    for env in ENVIRONMENTS:
        runtime = load(figures_root, env, "runtime")
        entry = runtime[
            (runtime["algorithm"] == "transport")
            & (runtime["horizon"] == MAIN_HORIZON[env])
            & (runtime["flow"] == MAIN_FLOW[env])
        ]
        if entry.empty:
            continue
        budget = float(entry["simulator_budget_mean"].iloc[0])
        main, auxiliary = TRANSPORT_ALLOCATION[env]
        dimension = free_coordinates[env]
        lines.append(
            " & ".join(
                [
                    DISPLAY_NAME[env],
                    argument[env],
                    str(dimension),
                    str(main),
                    str(auxiliary),
                    f"${auxiliary / dimension:.1f}$",
                    f"${budget:,.0f}$".replace(",", "\\,"),
                ]
            )
            + " \\\\"
        )
    caption = (
        "Split of the matched simulator budget between the main trajectories $B_{\\mathrm{tr}}$ and the auxiliary "
        "sensitivity trajectories $n_{\\mathrm{tr}}$ of the fixed-scale transport estimator, with $d$ the number of "
        "free coordinates of the population argument. The equal-budget rule constrains $T(B_{\\mathrm{tr}}+"
        "n_{\\mathrm{tr}})$ but not the split, which was reallocated towards the auxiliary estimate on the "
        "continuous-state benchmarks and left at the default on the finite-state ones. The resulting auxiliary "
        "sample size per coordinate spans more than two orders of magnitude, and distribution planning is the one "
        "benchmark where it falls close to one."
    )
    return table_environment("\n".join(lines), caption, "tab:auxiliary-budget", "llrrrrr")


def flow_pivot_header(horizons, flows):
    """Header putting the population-flow modes side by side under each horizon."""
    if len(flows) == 1:
        return ["Estimator & " + " & ".join(f"$T={h}$" for h in horizons) + " \\\\", "\\midrule"]
    span = len(flows)
    top = "Estimator" + "".join(f" & \\multicolumn{{{span}}}{{c}}{{$T={h}$}}" for h in horizons) + " \\\\"
    rules = " ".join(
        f"\\cmidrule(lr){{{2 + i * span}-{1 + (i + 1) * span}}}" for i in range(len(horizons))
    )
    sub = " & " + " & ".join(f.capitalize() for _ in horizons for f in flows) + " \\\\"
    return [top, rules, sub, "\\midrule"]


def flow_pivot_rows(grouped, horizons, flows, cell):
    """One row per estimator, with a cell for every (horizon, flow) pair."""
    lines = []
    for label, rows in grouped.groupby("label", sort=False):
        cells = []
        for horizon in horizons:
            for flow in flows:
                entry = rows[(rows["horizon"] == horizon) & (rows["flow"] == flow)]
                cells.append("---" if entry.empty else cell(entry.iloc[0]))
        lines.append(" & ".join([pretty_label(label)] + cells) + " \\\\")
    return lines


def full_objectives(figures_root, env):
    """Appendix table: every configuration of one benchmark."""
    grouped = grouped_objectives(figures_root, env)
    grouped = grouped.sort_values(["label", "horizon", "flow"])
    horizons = sorted(grouped["horizon"].unique())
    flows = sorted(grouped["flow"].unique())
    lines = flow_pivot_header(horizons, flows)
    lines += flow_pivot_rows(grouped, horizons, flows, lambda row: with_error(row, 4))
    caption = (
        f"{DISPLAY_NAME[env]} benchmark: final validation objective for every configuration, as mean and "
        "standard deviation over five seeds. Higher is better."
    )
    if len(flows) > 1:
        caption += " The exact and particle population flows are shown side by side at each horizon."
    return table_environment(
        "\n".join(lines),
        caption,
        f"tab:full-objectives-{env}",
        "l" + "r" * (len(horizons) * len(flows)),
        size="\\footnotesize\n\\setlength{\\tabcolsep}{4pt}" if len(horizons) * len(flows) > 2 else None,
    )


def tv_bounds(figures_root):
    """Appendix table: the total-variation perturbation bound along the learned flows."""
    lines = [
        "Benchmark & Configurations & Measured points & $\\max_t d_{\\mathrm{TV}}/\\lambda$ & Satisfies $d_{\\mathrm{TV}}\\leq\\lambda$ \\\\",
        "\\midrule",
    ]
    total, satisfied, configurations = 0, 0, 0
    for env in ENVIRONMENTS:
        table = load(figures_root, env, "transport_tv_bounds")
        if table is None or table.empty:
            continue
        ratio = (table["max_tv_upper_bound"] / table["lambda"]).max()
        groups = table.groupby(["label", "flow", "horizon"]).ngroups
        total += len(table)
        configurations += groups
        satisfied += int(table["satisfies_lambda_bound"].sum())
        lines.append(
            " & ".join(
                [
                    DISPLAY_NAME[env],
                    str(groups),
                    str(len(table)),
                    f"${ratio:.3f}$",
                    f"{int(table['satisfies_lambda_bound'].sum())}/{len(table)}",
                ]
            )
            + " \\\\"
        )
    lines.append("\\midrule")
    lines.append(f"Total & ${configurations}$ & ${total}$ & & {satisfied}/{total} \\\\")
    caption = (
        "Perturbation bound on the finite-state benchmarks. For every transport run the largest value of "
        "$d_{\\mathrm{TV}}(M_t^{\\lambda,\\theta},\\mu_t^\\theta)$ along the learned population flow is compared "
        "with $\\lambda$. The bound holds at every measured point, and is attained to within a factor "
        "$d_{\\mathrm{TV}}(q_t,\\mu_t^\\theta)\\leq1$ of $\\lambda$, so the perturbation is neither degenerate "
        "nor larger than the theory allows."
    )
    return table_environment("\n".join(lines), caption, "tab:tv-bounds", "lrrrr")


def twostate_policy_errors(figures_root):
    """Appendix table: full policy-recovery breakdown."""
    table = load(figures_root, "twostate", "policy_error")
    grouped = (
        table.groupby(["label", "flow", "horizon"])[["mean_abs_policy_error", "max_abs_policy_error"]]
        .mean()
        .reset_index()
        .sort_values(["flow", "label"])
    )
    grouped = grouped.sort_values(["label", "horizon", "flow"])
    horizons = sorted(grouped["horizon"].unique())
    flows = sorted(grouped["flow"].unique())
    lines = flow_pivot_header(horizons, flows)
    lines += flow_pivot_rows(
        grouped, horizons, flows, lambda row: f"${float(row['mean_abs_policy_error']):.4f}$"
    )
    caption = (
        "Two-state benchmark: mean absolute deviation of the learned policy from the closed-form optimum, "
        "averaged over the two states and over five seeds, for every configuration. The exact and particle "
        "population flows are shown side by side at each horizon."
    )
    return table_environment(
        "\n".join(lines),
        caption,
        "tab:twostate-policy-error",
        "l" + "r" * (len(horizons) * len(flows)),
    )


def gradient_diagnostics_table(figures_root, env):
    """Measured bias, dispersion and mean-square error of the transport gradient."""
    table = load(figures_root, env, "gradient_diagnostics")
    if table is None or table.empty:
        return None
    table = table[table["flow"] == MAIN_FLOW[env]]
    if table.empty:
        return None

    lines = [
        "$\\lambda$ & $\\lVert\\nabla J^\\lambda-\\nabla J\\rVert$ & $/\\lambda^2$ "
        "& $\\lVert\\E[\\widehat G]-\\nabla J^\\lambda\\rVert$ & $p$ "
        "& Dispersion & $d\\cdot\\mathrm{MSE}$ & $\\cos$ \\\\",
        "\\midrule",
    ]
    for lambda_value, rows in table.groupby("lambda"):
        perturbation = rows["perturbation_bias_norm"].mean()
        bias = rows["bias_norm"].mean()
        dispersion = rows["estimate_std"].mean()
        dimension = int(rows["n_parameters"].iloc[0])
        lines.append(
            " & ".join(
                [
                    f"${lambda_value:g}$",
                    f"${perturbation:.4f}$",
                    f"${perturbation / lambda_value ** 2:.2f}$",
                    f"${bias:.4f}$",
                    f"${rows['bias_chi2_pvalue'].mean():.3f}$",
                    f"${dispersion:.4f}$",
                    f"${dimension * rows['mse'].mean():.4f}$",
                    f"${rows['cosine_similarity'].mean():.3f}$",
                ]
            )
            + " \\\\"
        )
    replications = int(table["n_replications"].iloc[0])
    particles = int(table["diagnostic_n_particles"].iloc[0])
    caption = (
        f"Evaluation-only gradient diagnostics on the {DISPLAY_NAME[env].lower()} benchmark, at the policies saved "
        f"at the end of training and averaged over the five seeds, from {replications} independent replications of "
        f"the estimator at {particles} particles. Column two is the perturbation bias of the gradient, "
        "$\\lVert\\nabla_\\theta J^\\lambda(\\theta)-\\nabla_\\theta J(\\theta)\\rVert$, computed from the "
        "closed-form gradient oracle; column three divides it by $\\lambda^2$. Column four is the bias of the "
        "estimator with respect to the perturbed gradient it targets, and column five the $p$-value of the "
        "$\\chi^2$ test that this bias is zero at the available replication count, so a large $p$ means the "
        "estimator is not measurably biased for $\\nabla_\\theta J^\\lambda$. Column six is the dispersion "
        "$\\lVert\\operatorname{sd}(\\widehat G)\\rVert$ of the replications and column seven the mean-square "
        "error scaled by the parameter dimension $d$; it equals the sum of the squares of columns four and six, up to the factor $(R-1)/R$ carried by the unbiased dispersion estimate, and the identity is verified run by run to $2\\%$, which is exactly that factor at $R=50$. "
        "The last column is the cosine alignment of the mean estimate with the oracle gradient."
    )
    return table_environment("\n".join(lines), caption, f"tab:gradient-diagnostics-{env}", "lrrrrrrr", size="\\small")


def objective_bias_scaling(figures_root):
    """Perturbation bias of the optimal value against the perturbation scale."""
    lines = [
        "Benchmark & $\\lambda$ & $J^\\lambda(\\theta_\\lambda^\\star)$ "
        "& $|J^\\lambda(\\theta_\\lambda^\\star)-J(\\theta^\\star)|$ & $/\\lambda$ & $/\\lambda^2$ \\\\",
        "\\midrule",
    ]
    for env in ["lq", "portfolio"]:
        table = load(figures_root, env, "objectives")
        table = table[(table["flow"] == MAIN_FLOW[env]) & (table["label"].str.startswith("Transport"))].copy()
        sign = 1.0 if table["objective_convention"].dropna().iloc[0] == "cost" else -1.0
        table["lambda"] = table["label"].str.extract(r"lambda=([\d.]+)").astype(float)
        first = True
        for lambda_value, rows in table.groupby("lambda"):
            bias = sign * (rows["Jlambda_star"].iloc[0] - rows["J0_star"].iloc[0])
            lines.append(
                " & ".join(
                    [
                        DISPLAY_NAME[env] if first else "",
                        f"${lambda_value:g}$",
                        f"${rows['Jlambda_star'].iloc[0]:.3f}$",
                        f"${bias:.4f}$",
                        f"${bias / lambda_value:.2f}$",
                        f"${bias / lambda_value ** 2:.2f}$",
                    ]
                )
                + " \\\\"
            )
            first = False
        if env != "portfolio":
            lines.append("\\midrule")
    caption = (
        "Perturbation bias of the optimal value on the two benchmarks where the perturbed objective and its "
        "optimizer are both available in closed form. Theorem~\\ref{theorem:discrete-convergence-objective} bounds "
        "this quantity by $C_T\\lambda$ in general, so the ratio in the fifth column is bounded and the sixth "
        "column would diverge if the rate were exactly linear. It does not: the sixth column is constant to within $0.1\\%$ "
        "over a sixteen-fold range of $\\lambda$ on the linear--quadratic benchmark, and "
        "constant to within $7\\%$ on the portfolio benchmark. Both benchmarks are quadratic in the population "
        "argument, for which the linear term of the expansion vanishes and the bias is $O(\\lambda^2)$; the "
        "measurement recovers that rate rather than the generic bound."
    )
    return table_environment("\n".join(lines), caption, "tab:objective-bias-scaling", "llrrrr", size="\\small")


def reallocation(figures_root, results_root=None):
    """Default against reallocated split of the same simulator budget.

    The equal-budget rule constrains only T(B+n), so the split can be moved towards
    the auxiliary sensitivity estimate at no cost in simulated transitions. These runs
    do exactly that on the two finite-state benchmarks whose default split is smallest
    relative to the dimension of the population argument.
    """
    import sys

    results_root = Path(results_root or ROOT / "results")
    sys.path.insert(0, str(ROOT / "src"))
    from mfc.visualization import load_runs
    from mfc.visualization.io import validation_dataframe

    # (benchmark, lambda, default split, reallocated split, iteration at which both are read)
    settings = [
        ("distribution", 0.4, (549, 11), (200, 360), 5_000),
        ("advertising", 0.1, (248, 12), (100, 160), 10_000),
    ]
    lines = [
        "Benchmark & $\\lambda$ & $B_{\\mathrm{tr}}$ & $n_{\\mathrm{tr}}$ & $n_{\\mathrm{tr}}/d$ "
        "& Iterations & Objective & Seeds \\\\",
        "\\midrule",
    ]
    dimensions = {"distribution": 9, "advertising": 1}
    any_row = False
    for env, lambda_value, default, reallocated, checkpoint in settings:
        rows = []
        for split, root in [(default, results_root), (reallocated, results_root / "aux_budget")]:
            runs = [
                run
                for run in load_runs(root, env=env)
                if run["metadata"]["algorithm"] == "transport"
                and run["metadata"]["perturbation"] == lambda_value
                and run["metadata"]["flow"] == "exact"
            ]
            if not runs:
                continue
            curves = validation_dataframe(runs)
            at_checkpoint = curves[curves["step"] == checkpoint]
            if at_checkpoint.empty:
                continue
            values = at_checkpoint.groupby("seed")["validation_reward"].last()
            rows.append((split, values.mean(), values.std(), len(values)))
        if len(rows) < 2:
            continue
        any_row = True
        for index, (split, mean, deviation, count) in enumerate(rows):
            lines.append(
                " & ".join(
                    [
                        DISPLAY_NAME[env] if index == 0 else "",
                        f"${lambda_value:g}$" if index == 0 else "",
                        str(split[0]),
                        str(split[1]),
                        f"${split[1] / dimensions[env]:.1f}$",
                        (f"${checkpoint:,d}$".replace(",", "\\,") if index == 0 else ""),
                        f"${mean:.4f}\\pm{deviation:.4f}$",
                        str(count),
                    ]
                )
                + " \\\\"
            )
        lines.append("\\midrule")
    if not any_row:
        return None
    lines = lines[:-1]

    caption = (
        "Recorded check of moving the matched simulator budget from the main trajectories towards the auxiliary "
        "sensitivity estimate. Both rows use the same estimator, perturbation scales, simulator budget "
        "$T(B_{\\mathrm{tr}}+n_{\\mathrm{tr}})$ and number of training iterations; only the split differs. "
        "The table compares the two allocations at the iteration count given in the sixth column, reading the "
        "default arm from its recorded validation history at the same point, and reports the mean and standard "
        "deviation over the seeds indicated. Higher is better."
    )
    return table_environment("\n".join(lines), caption, "tab:reallocation", "llrrrrrr", size="\\small")


def parse_args():
    parser = argparse.ArgumentParser(description="Emit the LaTeX tables of the numerical-experiments chapter.")
    parser.add_argument("--figures-root", default=str(ROOT / "results" / "figures"))
    parser.add_argument("--output-root", default=str(ROOT / "files" / "report" / "tables"))
    return parser.parse_args()


def main():
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    tables = {
        "objective_summary": objective_summary(args.figures_root),
        "budget_runtime": budget_runtime(args.figures_root),
        "twostate_horizon": twostate_horizon(args.figures_root),
        "twostate_grid": twostate_grid(args.figures_root),
        "lq_decomposition": decomposition(args.figures_root, "lq"),
        "portfolio_decomposition": decomposition(args.figures_root, "portfolio"),
        "adaptive_scales": adaptive_scales(args.figures_root),
        "tv_bounds": tv_bounds(args.figures_root),
        "twostate_policy_error": twostate_policy_errors(args.figures_root),
        "auxiliary_budget": auxiliary_budget(args.figures_root),
        "objective_bias_scaling": objective_bias_scaling(args.figures_root),
    }
    reallocation_table = reallocation(args.figures_root)
    if reallocation_table is not None:
        tables["reallocation"] = reallocation_table
    tables |= {
    }
    for env in ENVIRONMENTS:
        tables[f"full_objectives_{env}"] = full_objectives(args.figures_root, env)
        if env in {"distribution", "cybersecurity"}:
            # The other benchmarks carry a more specific table of their own.
            tables[f"summary_{env}"] = benchmark_summary(args.figures_root, env)

    for env in ["lq", "portfolio"]:
        diagnostics = gradient_diagnostics_table(args.figures_root, env)
        if diagnostics is not None:
            tables[f"gradient_diagnostics_{env}"] = diagnostics

    for name, body in tables.items():
        path = output_root / f"{name}.tex"
        path.write_text(body)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
