import logging
import multiprocessing
import pickle
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches

from scripts.convergence_criterion import ConvergenceCriterion
from scripts.create_plots import create_plots
from util.tensorboard_logs import create_event_accumulators, read_scalar

# Disable the loggers for the imported scripts (since these just spam too much)
logging.basicConfig(level=logging.CRITICAL)

from tqdm import tqdm

from run_configs import RUN_CONFIGS

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

log_dir = Path("/is", "ei", "jschneider", "action_space_toolbox_logs", "training")
out_dir = Path(__file__).parents[2] / "out"


def lighten(colors):
    results = []
    for color in colors:
        results_ints = [
            min(int(int(f"0x{c}", 16) * 1.2), 255)
            for c in [color[1:3], color[3:5], color[5:7]]
        ]
        results.append(
            f"#{hex(results_ints[0])[2:]}{hex(results_ints[1])[2:]}{hex(results_ints[2])[2:]}"
        )
    return results


GREY = "#AAAAAA"
LIGHT_GREY = lighten([GREY])[0]

COLORSCHEME = [
    "#648FFF",
    "#785EF0",
    "#DC267F",
]
COLORSCHEME_LIGHT = lighten(COLORSCHEME)

GOLDEN_RATIO = 1.618


def env_name_to_diplay(name):
    if name == "dmc_Finger-spin_TC-v1":
        return "Finger-spin"
    elif name == "Walker2d_TC-v3":
        return "Walker2D"


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
    run_dirs = [d for d in log_path.iterdir() if d.is_dir() and d.name.isnumeric()]
    if len(run_dirs) > 0:
        tb_dirs = [run_dir / "tensorboard" for run_dir in run_dirs]
        event_accumulators = [ea for _, ea in create_event_accumulators(tb_dirs)]
    scalars_value = []
    scalars_reward = []
    for ea in event_accumulators:
        try:
            scalars_curr_ea_value = read_scalar(ea, key)
            scalars_value.append(scalars_curr_ea_value)
        except:
            logger.warning(f"Did not find key {key} in all logs.")
        scalars_curr_ea_reward = read_scalar(ea, "rollout/ep_rew_mean")
        scalars_reward.append(scalars_curr_ea_reward)
    scalars_value = sorted(
        [list(s.values()) for s in scalars_value], key=lambda s: len(s)
    )
    scalars_reward = sorted(
        [list(s.values()) for s in scalars_reward], key=lambda s: len(s)
    )
    steps_value = set([s.step for s in scalars_value[-1]])
    for scalar in scalars_value:
        scalar.sort(key=lambda s: s.step)
        assert np.all(s.step in steps_value for s in scalar), "Steps do not match."
    if len(scalars_value[0]) < len(scalars_value[-1]):
        logger.warning(
            f"Found a different number of scalars for the event accumulators (key: {key}, "
            f"min: {len(scalars_value[0])}, max: {len(scalars_value[-1])}), using the "
            f"{'minimum' if only_complete_steps else 'maximum'} value"
        )
    if only_complete_steps:
        steps_value = [scalar.step for scalar in scalars_value[0]]
    else:
        steps_value = [scalar.step for scalar in scalars_value[-1]]
    if only_complete_steps:
        steps_reward = [scalar.step for scalar in scalars_reward[0]]
    else:
        steps_reward = [scalar.step for scalar in scalars_reward[-1]]

    steps_value = [step for step in steps_value if step <= max_step]
    steps_reward = [step for step in steps_reward if step <= max_step]

    values = [
        [
            scalars_curr_run[i].value
            for scalars_curr_run in scalars_value
            if i < len(scalars_curr_run)
        ]
        for i in range(len(steps_value))
    ]
    rewards = [
        [
            scalars_curr_run[i].value
            for scalars_curr_run in scalars_reward
            if i < len(scalars_curr_run)
        ]
        for i in range(len(steps_reward))
    ]
    return (steps_value, values), (steps_reward, rewards)


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
            data["low_sample/estimated_gradient"][0]["mean"],
            data["true_gradient"][1]["mean"],
            data["estimated_gradient"][1]["mean"],
            data["low_sample/estimated_gradient"][1]["mean"],
            data["true_gradient"][2]["mean"],
            data["estimated_gradient"][2]["mean"],
            data["low_sample/estimated_gradient"][2]["mean"],
        ],
        yerr=[
            data["true_gradient"][0]["std"],
            data["estimated_gradient"][0]["std"],
            data["low_sample/estimated_gradient"][0]["std"],
            data["true_gradient"][1]["std"],
            data["estimated_gradient"][1]["std"],
            data["low_sample/estimated_gradient"][1]["std"],
            data["true_gradient"][2]["std"],
            data["estimated_gradient"][2]["std"],
            data["low_sample/estimated_gradient"][2]["std"],
        ],
        error_kw=error_bar_props,
        color=[c for cp in zip(COLORSCHEME, COLORSCHEME_LIGHT) for c in cp],
        edgecolor=[c for c in COLORSCHEME for _ in range(2)],
        width=1,
    )
    bar_xpos += 10
    for i, bar in enumerate(bars):
        if i % 3 == 1:
            bar.set_hatch("///")
        elif i % 3 == 2:
            bar.set_hatch("XXX")


def create_plots_iclr_gradient_subspace_fraction():
    global bar_xpos
    cache_file = Path(
        "/home/jschneider/Seafile/PhD/project_optimal_action_spaces/iclr_2024/gradient_subspace_fraction_cache.pkl"
    )
    if not cache_file.exists() or True:
        results = {}
        for env_name, run_config in RUN_CONFIGS.items():
            if env_name in "dmc_Finger-spin_TC-v1" or env_name in "Walker2d_TC-v3":
                results[env_name] = {}
                curr_results_env = results[env_name]
                for algorithm_name, algorthm_log_path in run_config["log_dirs"].items():
                    curr_results_env[algorithm_name] = {}
                    curr_results_algo = curr_results_env[algorithm_name]
                    algorthm_log_path = log_dir / env_name / algorthm_log_path
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
                            ) = load_logs(algorthm_log_path, key, run_config["xmax"])
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

        with cache_file.open("wb") as f:
            pickle.dump(results, f)
    else:
        with cache_file.open("rb") as f:
            results = pickle.load(f)

    for loss_type in ["policy", "vf"]:
        bar_xpos = 0
        factor = 2
        # fig, ax = plt.subplots(figsize=(factor * GOLDEN_RATIO * 4, factor))
        plt.rc("font", size=8)
        fig, ax = plt.subplots()

        # Calculate width and height for the desired aspect ratio
        width = 0.9  # This value might need adjustment
        height = width / 3.24

        # Position the axes
        left = 0.15  # Leave space for y-axis labels
        bottom = 0.5 - (height / 2)  # Center the axes in the figure
        ax.set_position([left, bottom, width, height])
        for env_name in ["dmc_Finger-spin_TC-v1", "Walker2d_TC-v3"]:
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
                (x_pos_start_env + bar_xpos - 4) / 2 - 0.5,
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
        ax.spines["left"].set_bounds(0, 1.0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_ylabel("Gradient subspace fraction")
        legend_handles = []
        legend_handles.append(
            patches.Patch(color=COLORSCHEME[0], label="Initial phase")
        )
        legend_handles.append(
            patches.Patch(color=COLORSCHEME[1], label="Training phase")
        )
        legend_handles.append(patches.Patch(color=COLORSCHEME[2], label="Convergence"))
        legend_handles.append(
            patches.Rectangle(
                (0, 0),
                1,
                1,
                fc="w",
                fill=False,
                edgecolor="none",
                linewidth=0,
                alpha=0,
            )
        )
        legend_handles.append(
            patches.Patch(color=GREY, label="True gradient, true Hessian")
        )
        legend_handles.append(
            patches.Patch(
                facecolor=LIGHT_GREY,
                edgecolor=GREY,
                hatch="///",
                label="Approx. gradient",
            )
        )
        legend_handles.append(
            patches.Patch(
                facecolor=LIGHT_GREY,
                edgecolor=GREY,
                hatch="XXX",
                label="Approx. gradient, approx. Hessian",
            )
        )
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
        out_path = Path(
            f"/home/jschneider/Seafile/PhD/project_optimal_action_spaces/iclr_2024/gradient_subspace_fraction/gradient_subspace_fraction_{loss_type}"
        )
        # legend_fig.savefig(
        #     out_path.parent / (out_path.name + "_legend.pdf"), bbox_inches=bbox
        # )
        fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")


if __name__ == "__main__":
    create_plots_iclr_gradient_subspace_fraction()
