import logging
import multiprocessing
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

from scripts.create_plots import create_plots

# Disable the loggers for the imported scripts (since these just spam too much)
logging.basicConfig(level=logging.CRITICAL)

from tqdm import tqdm

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

log_dir = Path("/is", "ei", "jschneider", "action_space_toolbox_logs", "training")
out_dir = Path(__file__).parents[2] / "out"
RUN_CONFIGS = {
    "Ant_TC-v3": {
        "log_dirs": {"ppo": "2023-09-22/18-15-58", "sac": "2023-09-26/16-54-05"},
        "xmax": 2_000_000,
    },
    "HalfCheetah_TC-v3": {
        "log_dirs": {"ppo": "2023-07-14/21-58-53", "sac": "2023-09-19/11-08-06"},
        "xmax": 3_000_000,
        "analysis_run_ids": {"ppo": "repeat_low_sample", "sac": "repeat_after_bugfix"},
    },
    "Pendulum_TC-v1": {
        "log_dirs": {"ppo": "2023-09-22/10-51-27", "sac": "2023-09-22/19-13-25"},
        "xmax": 300_000,
    },
    # "Pendulum_PC-v1": {"log_dirs": {"ppo": "2022-11-18/17-24-43/0"}, "xmax": 300_000},
    # "Pendulum_VC-v1": {"log_dirs": {"ppo": "2022-11-21/19-55-23/0"}, "xmax": 300_000},
    # "Reacher_PC-v2": {"log_dirs": {"ppo": "2022-11-14/13-57-50/0"}, "xmax": 1_000_000},
    # "Reacher_VC-v2": {"log_dirs": {"ppo": "2023-01-13/16-01-01/0"}, "xmax": 1_000_000},
    "Walker2d_TC-v3": {
        "log_dirs": {"ppo": "2023-07-14/23-14-41", "sac": "2023-09-23/23-20-09"},
        "xmax": 2_000_000,
    },
    "dmc_Cheetah-run_TC-v1": {
        "log_dirs": {"ppo": "2022-11-08/18-05-00"},
        "xmax": 3_000_000,
    },
    "dmc_Ball_in_cup-catch_TC-v1": {
        "log_dirs": {"ppo": "2023-09-22/10-49-35", "sac": "2023-09-24/20-17-17"},
        "xmax": 1_000_000,
    },
    "dmc_Finger-spin_TC-v1": {
        "log_dirs": {"ppo": "2022-12-21/20-44-24", "sac": "2023-09-21/10-34-29"},
        "xmax": 1_000_000,
    },
    "dmc_Walker-walk_TC-v1": {
        "log_dirs": {"ppo": "2022-11-09/17-48-20"},
        "xmax": 3_000_000,
    },
}
PLOT_CONFIGS = {
    "learning_curve_train": {
        "out_dir": "learning_curves_train",
        "key": "rollout/ep_rew_mean",
        "ylabel": "Cumulative reward",
        "smoothing_weight": 0.6,
    },
    "learning_curve_eval": {
        "out_dir": "learning_curves_eval",
        "key": "eval/mean_reward",
        "ylabel": "Cumulative reward",
        "smoothing_weight": 0.0,
    },
}


def worker(
    log_paths: Sequence[Path],
    legend: Optional[Sequence[str]],
    title: str,
    xlabel: str,
    ylabel: str,
    xlimits: Tuple[float, float],
    ylimits: Tuple[float, float],
    xaxis_log: bool,
    key: str,
    smoothing_weight: float,
    only_complete_steps: bool,
    separate_legend: bool,
    num_same_color_plots: int,
    fontsize: int,
    linewidth: float,
    out: Path,
):
    from scripts.create_multiline_plots import create_multiline_plots

    try:
        create_plots(
            log_paths,
            legend,
            title,
            xlabel,
            ylabel,
            xlimits,
            ylimits,
            xaxis_log,
            [key],
            smoothing_weight,
            only_complete_steps,
            separate_legend,
            num_same_color_plots,
            fontsize,
            linewidth,
            {},
            out,
        )
    except Exception as e:
        logger.warning(
            f"Could not create plot {out.relative_to(out.parents[2])}, got exception {e}"
        )


if __name__ == "__main__":
    results = []
    with multiprocessing.Pool(20) as pool:
        for env_name, run_config in RUN_CONFIGS.items():
            env_file_name = env_name[:-3].lower().replace("-", "_")
            if not env_name.startswith("dmc"):
                env_file_name = "gym_" + env_file_name
            for out_filename, plot_config in PLOT_CONFIGS.items():
                out_path = (
                    out_dir
                    / plot_config["out_dir"]
                    / f"{env_file_name}_{out_filename}.pdf"
                )
                out_path.parent.mkdir(exist_ok=True, parents=True)
                title_env_name = (
                    env_name[4:-3] if env_name.startswith("dmc_") else env_name[:-3]
                )
                results.append(
                    pool.apply_async(
                        worker,
                        (
                            [
                                Path(log_dir, env_name, experiment_dir)
                                for experiment_dir in run_config["log_dirs"].values()
                            ],
                            [
                                algo_name.upper()
                                for algo_name in run_config["log_dirs"].keys()
                            ],
                            plot_config.get("title"),
                            plot_config.get("xlabel", "Environment steps"),
                            plot_config["ylabel"],
                            (run_config.get("xmin", 0), run_config.get("xmax")),
                            (plot_config.get("ymin"), plot_config.get("ymax")),
                            False,
                            plot_config["key"],
                            plot_config["smoothing_weight"],
                            True,
                            True,
                            plot_config.get("num_same_color_plots", 1),
                            plot_config.get("fontsize", 22),
                            plot_config.get("linewidth", 1.5),
                            out_path,
                        ),
                    )
                )
        for res in tqdm(results):
            res.get()
