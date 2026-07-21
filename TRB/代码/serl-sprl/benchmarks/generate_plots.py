import importlib.util
import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scikit_posthocs as sp
import seaborn as sns
import yaml
from scipy import stats

import wandb

os.environ["PATH"] = "/usr/local/texlive/2025/bin/x86_64-linux:" + os.environ["PATH"]
mpl.rcParams["text.usetex"] = True
mpl.rcParams["text.latex.preamble"] = r"\usepackage{times}"
mpl.rcParams["font.family"] = "serif"
mpl.rcParams["font.size"] = 13


def find_safe_folder(base_path):
    """Find the Safe/safe folder in base_path, case-insensitively"""
    base_dir = Path(base_path)
    safe_folders = [f for f in base_dir.iterdir() if f.is_dir() and f.name.lower() == "safe"]

    if safe_folders:
        return safe_folders[0]
    else:
        raise FileNotFoundError(f"No 'safe' folder found in {base_path}")


def compute_conf_intervals(all_returns):
    all_ci_low = []
    all_ci_high = []
    for g in all_returns:
        ci = stats.bootstrap(
            np.array(g).reshape(1, -1),
            n_resamples=100,
            confidence_level=0.95,
            random_state=rng,
            statistic=trimmed_mean,
        )
        (
            all_ci_low.append(ci.confidence_interval.low)
            if not np.isnan(ci.confidence_interval.low)
            else all_ci_low.append(0.0)
        )
        (
            all_ci_high.append(ci.confidence_interval.high)
            if not np.isnan(ci.confidence_interval.high)
            else all_ci_high.append(0.02)
        )
    results = np.stack([interquartile_means, all_ci_low, all_ci_high], axis=1)
    results_df = pd.DataFrame(
        results,
        columns=["interquartile_mean", "ci_lower", "ci_upper"],
        index=["SE-RL", "SE-RL Penalty", "SP-RL", "SP-RL PSL", "SP-RL PenC"],
    )
    if policy_type == "stochastic":
        results_df.loc["SP-RL"] = results_df.loc["SE-RL"]
        results_df.loc["SP-RL PenC"] = results_df.loc["SE-RL Penalty"]
    results_df.to_csv(os.getcwd() + f"/benchmarks/plots/{approach}/{environment}/{policy_type}_iqm.csv", index=False)
    return results_df


def plot_iqm(dataframes, titles, policy_type):
    n_tasks = len(dataframes)
    n_algos = len(dataframes[0])
    lower_bounds = [-30, -70, -60]
    upper_bounds = [0, -37, -5]

    fig, axes = plt.subplots(1, n_tasks, figsize=(4 * n_tasks, 3), sharey=True)

    if n_tasks == 1:
        axes = [axes]

    algo_names = dataframes[0].index.tolist()
    y_pos = np.array([0, 1, 2.5, 3.5, 4.5])  # Add extra space between index 1 and 2

    # Generate a color palette
    palette = sns.color_palette("colorblind", n_algos)
    algo_colors = {algo: palette[i] for i, algo in enumerate(algo_names)}

    box_height = 0.6  # Thickness of each box

    for task_num, (ax, df, title) in enumerate(zip(axes, dataframes, titles)):
        means = df["interquartile_mean"].values
        lowers = df["ci_lower"].values
        uppers = df["ci_upper"].values

        for i, algo in enumerate(algo_names):
            lower = lowers[i]
            upper = uppers[i]
            mean = means[i]
            if mean < lower_bounds[task_num]:
                mean = lower_bounds[task_num] + 0.5  # Adjust mean to be visible
                lower = lower_bounds[task_num] + 0.25
                upper = lower_bounds[task_num] + 0.75
            color = algo_colors[algo]

            # Draw box from lower to upper bound
            rect = patches.Rectangle(
                (lower, y_pos[i] - box_height / 2),  # (x, y) bottom left
                upper - lower,  # width
                box_height,  # height
                facecolor=color,
                edgecolor=color,
                alpha=0.5,
                linewidth=1.5,
            )
            ax.add_patch(rect)

            # Draw a vertical line at the mean
            ax.plot([mean, mean], [y_pos[i] - box_height / 2, y_pos[i] + box_height / 2], color="black", linewidth=1.5)

        # Add a dashed horizontal line to separate first two from last three algorithms
        separator_y = 1.75  # Between SE-RL Penalty (y=1) and SP-RL (y=2.5)
        ax.axhline(y=separator_y, color="gray", linestyle="--", linewidth=1.0, alpha=0.7)

        ax.set_xlim(lower_bounds[task_num], upper_bounds[task_num])
        ax.set_title(title)
        ax.set_xlabel("Interquartile Mean")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(algo_names)
        ax.invert_yaxis()

    plt.tight_layout()
    plt.show()
    fig.savefig(os.getcwd() + f"/benchmarks/plots/iqm_plot_{policy_type}.pdf", bbox_inches="tight", dpi=300)


def normalize_data(data, environment):
    if environment == "Seeker":
        low = -60  # just a lower bound
        high = 0  # actual best reward
    elif environment == "Pendulum":
        low = -30  # just a lower bound
        high = 0  # actual best reward
    elif environment == "Quadrotor":
        low = -70  # just a lower bound
        high = -5  # actual best reward
    data = np.clip(data, low, high)
    return (data - low) / (high - low)


def trimmed_mean(data, *args, **kwargs):
    return stats.trim_mean(data, 0.25, axis=0)


def write_csv(data, file):
    df = pd.DataFrame(
        data,
        columns=[
            "episode",
            "mean_return",
            "std_return",
            "ci_lower_return",
            "ci_upper_return",
            "mean_siv",
            "std_siv",
            "ci_lower_siv",
            "ci_upper_siv",
        ],
    )
    df.to_csv(file, index=False)


def process_combination(
    environment, approach, algorithm, improvement_strategy, w, run_ids, global_steps, stats_interval, rng
):
    path = f"/{approach}/{environment}/{algorithm}/{improvement_strategy}/{w}"
    file_path = os.getcwd() + f"/benchmarks/plots/{path}/"
    os.makedirs(file_path, exist_ok=True)
    num_runs = len(run_ids)
    api = wandb.Api()
    returns = np.zeros((num_runs, global_steps))
    interventions = np.zeros((num_runs, global_steps))

    for i in range(num_runs):
        run_str = "srl4ps/safe_gradients/" + run_ids[i]
        run = api.run(run_str)
        history = run.scan_history(
            keys=[
                "safety/return_without_pun",
                "safety/ep_mean_policy_interventions",
                "safety/ep_mean_interventions",
                "global_step",
            ]
        )

        if approach == "SERL":
            siv = [row["safety/ep_mean_interventions"] for row in history]
        else:
            siv = [row["safety/ep_mean_policy_interventions"] for row in history]
        ret = [row["safety/return_without_pun"] for row in history]
        glob_steps = [row["global_step"] for row in history]
        if len(ret) == 0:
            print(f"Run {run_str} has no data.")
            break
        for index, step in enumerate(glob_steps):
            if step >= global_steps:
                returns[i, global_steps - 1] = ret[index]
                interventions[i, global_steps - 1] = siv[index]
            else:
                returns[i, step - 1] = ret[index]
                interventions[i, step - 1] = siv[index]

    if len(ret) == 0:
        history = pd.read_csv(
            os.getcwd()
            + f"/benchmarks/plots/missing_results/{approach}_{environment}_{algorithm}_{improvement_strategy}_{w}.csv"
        )
        base_str = f"{environment}_{algorithm}_{improvement_strategy}"
        history_ret = history[[base_str + f"_seed_{i+1} - safety/return_without_pun" for i in range(num_runs)]]
        if approach == "SERL":
            history_siv = history[[base_str + f"_seed_{i+1} - safety/ep_mean_interventions" for i in range(num_runs)]]
        else:
            history_siv = history[
                [base_str + f"_seed_{i+1} - safety/ep_mean_policy_interventions" for i in range(num_runs)]
            ]
        for seed in range(num_runs):
            for index, step in enumerate(history["global_step"]):
                returns[seed, step - 1] = history_ret[base_str + f"_seed_{seed+1} - safety/return_without_pun"].iloc[
                    index
                ]
                if approach == "SERL":
                    interventions[seed, step - 1] = history_siv[
                        base_str + f"_seed_{seed+1} - safety/ep_mean_interventions"
                    ].iloc[index]
                else:
                    interventions[seed, step - 1] = history_siv[
                        base_str + f"_seed_{seed+1} - safety/ep_mean_policy_interventions"
                    ].iloc[index]
    # Convert arrays to pandas DataFrames for faster processing
    returns_df = pd.DataFrame(returns)
    interventions_df = pd.DataFrame(interventions)

    # Fill zero values with forward fill and backward fill
    returns_df = returns_df.replace(0, np.nan).ffill(axis=1).bfill(axis=1)
    interventions_df = interventions_df.replace(0, np.nan).ffill(axis=1).bfill(axis=1)

    # Convert back to numpy arrays
    returns = returns_df.to_numpy()
    interventions = interventions_df.to_numpy()

    # Downsample the arrays using stats_interval
    downsampled_steps = np.arange(0, global_steps, stats_interval)
    returns = returns[:, ::stats_interval]
    interventions = interventions[:, ::stats_interval]
    n_eps = len(downsampled_steps)

    # Compute means and standard deviations
    mean_returns = np.mean(returns, axis=0)
    std_returns = np.std(returns, axis=0)
    mean_siv = np.mean(interventions, axis=0)
    std_siv = np.std(interventions, axis=0)

    # Compute bootstrap confidence intervals
    n_resamples = 100
    confidence_level = 0.95

    # Initialize arrays for confidence intervals
    ci_returns = np.zeros((2, n_eps))
    ci_siv = np.zeros((2, n_eps))

    # Compute confidence intervals for each time step
    for step in range(n_eps):
        # Bootstrap for returns
        bootstrap_returns = stats.bootstrap(
            (returns[:, step],),
            n_resamples=n_resamples,
            confidence_level=confidence_level,
            random_state=rng,
            statistic=trimmed_mean,
        )
        ci_returns[:, step] = bootstrap_returns.confidence_interval

        # Bootstrap for interventions
        bootstrap_siv = stats.bootstrap(
            (interventions[:, step],),
            statistic=trimmed_mean,
            n_resamples=n_resamples,
            confidence_level=confidence_level,
            random_state=rng,
        )
        ci_siv[:, step] = bootstrap_siv.confidence_interval

    # Use downsampled steps for episodes
    eps = downsampled_steps

    # Stack all data including confidence intervals
    data = np.stack(
        [
            eps,
            mean_returns,
            std_returns,
            ci_returns[0],
            ci_returns[1],  # lower and upper CI for returns
            mean_siv,
            std_siv,
            ci_siv[0],
            ci_siv[1],
        ],  # lower and upper CI for interventions
        axis=1,
    )
    write_csv(data, file_path + "summary.csv")


if __name__ == "__main__":
    policy_type = "deterministic"  # "stochastic" or "deterministic"
    # Load configurations from best_runs.yml
    if policy_type == "deterministic":
        with open(os.getcwd() + "/benchmarks/best_runs.yml", "r") as f:
            configs = yaml.safe_load(f)
    else:
        with open(os.getcwd() + "/best_runs_stochastic.yml", "r") as f:
            configs = yaml.safe_load(f)

    stats_interval = 200
    num_seeds = 7
    p = 0.05  # Significance level for statistical tests

    deployment_results_path = os.getcwd() + "/results"
    environment_returns = {env: [] for env in configs["SERL"]}
    all_results = []

    rng = np.random.default_rng()

    # Process each combination from the config
    # ToDo: Get num of steps from .common of the respective subfolder
    for approach in configs:
        for environment in configs[approach]:
            # ToDo: Get num of steps from .common of the respective subfolder
            common_module_path = os.path.join(os.getcwd(), "benchmarks", environment.lower(), "common.py")
            spec = importlib.util.spec_from_file_location("common", common_module_path)
            common_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(common_module)
            global_steps = common_module.get_num_total_timesteps()

            for algorithm in configs[approach][environment]:
                for improvement_strategy in configs[approach][environment][algorithm]:
                    # Generate csv files for plotting training curves
                    for w, run_ids in configs[approach][environment][algorithm][improvement_strategy].items():
                        print(f"Processing {approach}/{environment}/{algorithm}/{improvement_strategy}/{w}")
                        # process_combination(
                        #     environment,
                        #     approach,
                        #     algorithm,
                        #     improvement_strategy,
                        #     w,
                        #     run_ids,
                        #     global_steps,
                        #     stats_interval,
                        #     rng
                        #     )

                        # Perform statistical tests on deployment results
                        approach_returns = []
                        # approach_goal_reached = []
                        for seed in range(num_seeds):
                            if improvement_strategy == "Vanilla":
                                base_dir = (
                                    deployment_results_path
                                    + f"/{approach}/{environment}/{algorithm}/{improvement_strategy}"
                                )
                                safe_folder = find_safe_folder(base_dir)
                            else:
                                base_dir = (
                                    deployment_results_path
                                    + f"/{approach}/{environment}/{algorithm}/{improvement_strategy}/{w}"
                                )
                                safe_folder = find_safe_folder(base_dir)
                            csv_path = safe_folder / f"_{seed+1}" / "deploy_logs.csv"
                            deployment_results = pd.read_csv(csv_path, sep=";")
                            returns = deployment_results["return_without_pun"]
                            approach_returns.extend(returns.tolist())
                        environment_returns[environment].append(approach_returns)

    for environment in environment_returns:
        all_returns = environment_returns[environment]
        statistics, p_value = stats.kruskal(*all_returns)
        interquartile_means = stats.trim_mean(all_returns, 0.25, axis=1)

        # Compute confidence intervals for interquartile means
        results_df = compute_conf_intervals(all_returns)
        all_results.append(results_df)
        # Print results
        print(f"Results for {environment}/{approach}/{algorithm}/{improvement_strategy}:")
        print(f"  Interquartile means: {interquartile_means}")
        print(f"Statistical test for {environment}/{approach}/{algorithm}/{improvement_strategy}:")
        print(f"  Kruskal-Wallis H-statistic: {statistics}, p-value: {p_value}")
        if p_value < p:
            print(f"  Significant difference found (p < {p}). Performing post-hoc tests...")
            posthoc = sp.posthoc_dunn(all_returns, p_adjust="bonferroni")
            print(posthoc)
            significant_pairs = posthoc[posthoc < p].stack().index.tolist()
            if significant_pairs:
                print("  Significant pairs:")
                for pair in significant_pairs:
                    print(f"    {pair[0]} vs {pair[1]}")
        else:
            print(f"  No significant difference (p >= {p})")

    # Plot interquartile means with confidence intervals
    plot_iqm(all_results, ["Pendulum", "Quadrotor", "Seeker"], policy_type)
