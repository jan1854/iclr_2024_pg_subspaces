import argparse
import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import yaml
from matplotlib import pyplot as plt
from matplotlib import patches

from pg_subspaces.metrics.read_metrics_cached import read_metrics_cached
from pg_subspaces.scripts.convergence_criterion import ConvergenceCriterion

# Disable the loggers for the imported scripts (since these just spam too much)
logging.basicConfig(level=logging.CRITICAL)

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

out_dir = Path(__file__).parents[2] / "out"


def lighten(colors, factor):
    results = []
    for color in colors:
        results_ints = [
            min(int(int(f"0x{c}", 16) * factor), 255)
            for c in [color[1:3], color[3:5], color[5:7]]
        ]
        results.append(
            f"#{hex(results_ints[0])[2:]}{hex(results_ints[1])[2:]}{hex(results_ints[2])[2:]}"
        )
    return results


GREY = "#AAAAAA"
LIGHT_GREY = lighten([GREY], 1.3)[0]
GREY = lighten([GREY], 0.95)[0]

COLORSCHEME = [
    "#648FFF",
    "#785EF0",
    "#DC267F",
]
COLORSCHEME_LIGHT = lighten(COLORSCHEME, 1.22)
COLORSCHEME = lighten(COLORSCHEME, 0.97)

GOLDEN_RATIO = 1.618


def env_name_to_diplay(name):
    name = name[:-3]
    if name.startswith("dmc_"):
        name = name[len("dmc_") :]
    if name == "Walker2d":
        return "Walker2D"
    if name == "Ball_in_cup-catch":
        return "Ball_in_cup"
    return name


def split_training_phases(
    steps_values: np.ndarray, values: np.ndarray, steps_rewards, rewards
):
    criterion = ConvergenceCriterion(20000, 0.1, 0.9)

    init, train, conv = criterion.split_steps(steps_rewards, rewards)
    end_init = init[0][-1]
    start_conv = conv[0][0]
    init_steps, init_values = zip(
        *[(s, v) for s, v in zip(steps_values, values) if s <= end_init]
    )
    train_steps, train_values = zip(
        *[
            (s, v)
            for s, v in zip(steps_values, values)
            if s > end_init and s < start_conv
        ]
    )
    conv_steps, conv_values = zip(
        *[(s, v) for s, v in zip(steps_values, values) if s >= start_conv]
    )

    return (
        (init_steps, train_steps, conv_steps),
        (init_values, train_values, conv_values),
    )


def load_logs(log_path, key, max_step, only_complete_steps=True):
    metrics = read_metrics_cached(log_path, [key])
    rewards = read_metrics_cached(log_path, ["rollout/ep_rew_mean"])

    metrics = sorted(metrics, key=lambda m: len(m[0]))
    rewards = sorted(rewards, key=lambda r: len(r[0]))
    if len(metrics[0][0]) < len(metrics[-1][0]):
        logger.warning(
            f"Found a different number of scalars for the event accumulators (key: {key}, "
            f"min: {len(metrics[0][0])}, max: {len(metrics[-1][0])}), using the "
            f"{'minimum' if only_complete_steps else 'maximum'} value"
        )
    if only_complete_steps:
        steps_metrics = metrics[0][0]
        steps_reward = rewards[0][0]
    else:
        steps_metrics = metrics[-1][0]
        steps_reward = rewards[-1][0]

    steps_metrics = [step for step in steps_metrics if step <= max_step]
    steps_reward = [step for step in steps_reward if step <= max_step]

    values_metrics = [
        [
            scalars_curr_run[1][i]
            for scalars_curr_run in metrics
            if i < len(scalars_curr_run[1])
        ]
        for i in range(len(steps_metrics))
    ]
    values_rewards = [
        [
            scalars_curr_run[1][i]
            for scalars_curr_run in rewards
            if i < len(scalars_curr_run[1])
        ]
        for i in range(len(steps_reward))
    ]
    return (steps_metrics, values_metrics), (steps_reward, values_rewards)


def plot_bars(data, ax):
    global bar_xpos
    error_bar_props = {
        "capsize": 2,  # Size of the horizontal caps
        "elinewidth": 1,  # Width of the error bars
        "capthick": 1,  # Thickness of the cap
        "ecolor": "grey",
    }
    bars = ax.bar(
        range(bar_xpos, bar_xpos + 9),
        [
            data["true_gradient"][0]["mean"],
            data["estimated_gradient"][0]["mean"],
            data["low_sample"][0]["mean"],
            data["true_gradient"][1]["mean"],
            data["estimated_gradient"][1]["mean"],
            data["low_sample"][1]["mean"],
            data["true_gradient"][2]["mean"],
            data["estimated_gradient"][2]["mean"],
            data["low_sample"][2]["mean"],
        ],
        yerr=[
            data["true_gradient"][0]["std"],
            data["estimated_gradient"][0]["std"],
            data["low_sample"][0]["std"],
            data["true_gradient"][1]["std"],
            data["estimated_gradient"][1]["std"],
            data["low_sample"][1]["std"],
            data["true_gradient"][2]["std"],
            data["estimated_gradient"][2]["std"],
            data["low_sample"][2]["std"],
        ],
        error_kw=error_bar_props,
        color=[
            c
            for cp in zip(COLORSCHEME_LIGHT, COLORSCHEME_LIGHT, COLORSCHEME_LIGHT)
            for c in cp
        ],
        edgecolor=[c for c in COLORSCHEME for _ in range(3)],
        width=0.85,
        zorder=2,
    )
    bar_xpos += 10
    for i, bar in enumerate(bars):
        if i % 3 == 1:
            bar.set_hatch("///")
        elif i % 3 == 2:
            bar.set_hatch("XXX")


def create_plots_iclr_gradient_subspace_fraction(
    log_dir, out_dir, env_names: Sequence[str]
):
    global bar_xpos
    results = {}
    with (Path(__file__).parent / "res" / "run_configs.yaml").open(
        "r"
    ) as run_configs_file:
        run_configs = yaml.safe_load(run_configs_file)
    for env_name, run_config in run_configs.items():
        if env_name in env_names:
            results[env_name] = {}
            curr_results_env = results[env_name]
            for algorithm_name, algorithm_log_path in run_config["log_dirs"].items():
                if algorithm_name in ["ppo", "sac"]:
                    curr_results_env[algorithm_name] = {}
                    curr_results_algo = curr_results_env[algorithm_name]
                    algorithm_log_path = log_dir / env_name / algorithm_log_path
                    for loss_type, loss_type_short in [
                        ("policy_loss", "policy"),
                        ("value_function_loss", "vf"),
                    ]:
                        curr_results_algo[loss_type_short] = {}
                        curr_results_loss = curr_results_algo[loss_type_short]
                        for grad_hess_type in [
                            "estimated_gradient",
                            "true_gradient",
                            "low_sample",
                        ]:
                            if grad_hess_type == "low_sample":
                                key = (
                                    f"high_curvature_subspace_analysis/{run_config.get('analysis_run_ids', {}).get(algorithm_name, 'default')}/low_sample/"
                                    f"gradient_subspace_fraction_100evs/estimated_gradient/{loss_type}"
                                )
                            else:
                                key = (
                                    f"high_curvature_subspace_analysis/{run_config.get('analysis_run_ids', {}).get(algorithm_name, 'default')}/"
                                    f"gradient_subspace_fraction_100evs/{grad_hess_type}/{loss_type}"
                                )
                            (steps_values, values), (
                                steps_rewards,
                                rewards,
                            ) = load_logs(algorithm_log_path, key, run_config["xmax"])
                            values_split = [[] for _ in range(3)]
                            for curr_values, curr_rewards, curr_steps in zip(
                                np.transpose(values),
                                np.transpose(rewards),
                                steps_values,
                            ):
                                _, curr_values_split = split_training_phases(
                                    steps_values,
                                    curr_values,
                                    steps_rewards,
                                    curr_rewards,
                                )
                                for i in range(len(values_split)):
                                    values_split[i].append(
                                        np.mean(curr_values_split[i])
                                    )

                            curr_results_loss[grad_hess_type] = [
                                {"mean": np.mean(v), "std": np.std(v)}
                                for v in values_split
                            ]

    for loss_type in ["policy", "vf"]:
        bar_xpos = 0
        # fig, ax = plt.subplots(figsize=(factor * GOLDEN_RATIO * 4, factor))
        plt.rc("font", size=7.5)
        fig, ax = plt.subplots()
        ax.set_zorder(10)
        ax.grid(axis="y", alpha=0.5)

        # Calculate width and height for the desired aspect ratio
        width = 0.9  # This value might need adjustment
        height = width / 3.24

        # Position the axes
        left = 0.15  # Leave space for y-axis labels
        bottom = 0.5 - (height / 2)  # Center the axes in the figure
        ax.set_position([left, bottom, width, height])
        print(results)
        for env_name in env_names:
            env_results = results[env_name]
            x_pos_start_env = bar_xpos
            for algorithm_name in ["ppo", "sac"]:
                x_pos_start_algo = bar_xpos
                plot_bars(env_results[algorithm_name][loss_type], ax)
                plt.text(
                    (x_pos_start_algo + bar_xpos - 2) / 2 - 0.5,
                    -0.1,
                    algorithm_name.upper(),
                    horizontalalignment="center",
                )
            bar_xpos += 1

            plt.text(
                (x_pos_start_env + bar_xpos - 2) / 2 - 0.5,
                -0.2,
                env_name_to_diplay(env_name),
                horizontalalignment="center",
            )

        ax.tick_params(
            axis="x",
            which="both",
            bottom=False,
            top=False,
            labelbottom=False,
        )
        ax.set_ylim(0, 1.02)
        ax.set_xlim(-1.3, bar_xpos - 1.7)
        ax.spines["left"].set_bounds(0, 1.0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_ylabel("Gradient subspace fraction")
        legend_handles = []
        legend_handles.append(
            patches.Patch(color=COLORSCHEME[0], label="Initial phase")
        )
        legend_handles.append(
            patches.Patch(
                facecolor=LIGHT_GREY,
                edgecolor=GREY,
                label="Precise gradient, precise Hessian",
            )
        )
        legend_handles.append(
            patches.Patch(color=COLORSCHEME[1], label="Training phase")
        )
        legend_handles.append(
            patches.Patch(
                facecolor=LIGHT_GREY,
                edgecolor=GREY,
                hatch="///",
                label="Mini-batch gradient, precise Hessian",
            )
        )
        legend_handles.append(
            patches.Patch(color=COLORSCHEME[2], label="Convergence phase")
        )
        legend_handles.append(
            patches.Patch(
                facecolor=LIGHT_GREY,
                edgecolor=GREY,
                hatch="XXX",
                label="Mini-batch gradient, mini-batch Hessian",
            )
        )
        # legend_handles.append(
        #     patches.Rectangle(
        #         (0, 0),
        #         1,
        #         1,
        #         fc="w",
        #         fill=False,
        #         edgecolor="none",
        #         linewidth=0,
        #         alpha=0,
        #     )
        # )
        # legend_plt = ax.legend(
        #     handles=legend_handles,
        #     frameon=False,
        #     ncol=len(legend_handles) // 2,
        #     bbox_to_anchor=(2.0, 2.0),
        # )
        # legend_fig = legend_plt.figure
        # legend_fig.canvas.draw()
        # bbox = legend_plt.get_window_extent().transformed(
        #     legend_fig.dpi_scale_trans.inverted()
        # )
        out_path = Path(out_dir / f"gradient_subspace_fraction_{loss_type}")
        # legend_fig.savefig(
        #     out_path.parent / (out_path.name + "_legend.pdf"), bbox_inches=bbox
        # )
        fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("log_dir", type=str)
    parser.add_argument("out_dir", type=str)
    parser.add_argument("env_names", type=str, nargs="+")
    args = parser.parse_args()
    create_plots_iclr_gradient_subspace_fraction(
        Path(args.log_dir), Path(args.out_dir), args.env_names
    )
