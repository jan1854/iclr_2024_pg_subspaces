import argparse
import logging
import multiprocessing
from pathlib import Path
from typing import Optional, Sequence, Tuple

import yaml
from tqdm import tqdm

from pg_subspaces.scripts.create_plots import create_plots

# Disable the loggers for the imported scripts (since these just spam too much)
logging.basicConfig(level=logging.CRITICAL)

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

out_dir = Path(__file__).parents[2] / "out"

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
            {},
            {},
            out,
        )
    except Exception as e:
        logger.warning(
            f"Could not create plot {out.relative_to(out.parents[2])}, got exception: {e}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("log_dir", type=str)
    parser.add_argument("envs", type=str, nargs="*", default=["*"])
    args = parser.parse_args()
    log_dir = Path(args.log_dir)
    with (Path(__file__).parent / "res" / "run_configs.yaml").open(
        "r"
    ) as run_configs_file:
        run_configs = yaml.safe_load(run_configs_file)
    with (Path(__file__).parent / "res" / "experiment_configs.yaml").open(
        "r"
    ) as experiment_configs_file:
        experiment_configs = yaml.safe_load(experiment_configs_file)

    results = []
    with multiprocessing.Pool(10) as pool:
        for env_name, run_config in run_configs.items():
            if env_name not in args.envs and "*" not in args.envs:
                continue
            env_file_name = env_name[:-3].lower().replace("-", "_")
            if not env_name.startswith("dmc"):
                env_file_name = "gym_" + env_file_name
            for out_experiment_dir, algorithms in experiment_configs.items():
                for (
                    out_filename,
                    plot_config,
                ) in PLOT_CONFIGS.items():
                    out_path = (
                        out_dir
                        / out_experiment_dir
                        / plot_config["out_dir"]
                        / f"{env_file_name}_{out_filename}"
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
                                    Path(
                                        log_dir,
                                        "training",
                                        env_name,
                                        run_config["log_dirs"][algo_name],
                                    )
                                    if algo_name in run_config["log_dirs"]
                                    else None
                                    for algo_name in algorithms.keys()
                                ],
                                [
                                    algo_descr
                                    for algo_name, algo_descr in algorithms.items()
                                    if algo_name in run_config["log_dirs"]
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
