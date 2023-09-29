import argparse
import logging
import multiprocessing
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

from scripts.create_plots import create_plots

# Disable the loggers for the imported scripts (since these just spam too much)
logging.basicConfig(level=logging.CRITICAL)

from tqdm import tqdm

from run_configs import RUN_CONFIGS

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

out_dir = Path(__file__).parents[2] / "out"
PLOT_CONFIGS = {
    "overlap_0100000_100evs": {
        "out_dir": "subspace_overlap",
        "key": "overlaps_top100_checkpoint0100000",
        "xlabel": "$t_2$",
        "ylabel": "Subspace overlap",
        "smoothing_weight": 0.0,
        "ymin": 0,
        "ymax": 1,
        "fill_in_data": {100000: 1.0},
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
    keys: str,
    smoothing_weight: float,
    only_complete_steps: bool,
    separate_legend: bool,
    num_same_color_plots: int,
    fontsize: int,
    linewidth: float,
    fill_in_data: Dict[int, float],
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
            keys,
            smoothing_weight,
            only_complete_steps,
            separate_legend,
            num_same_color_plots,
            fontsize,
            linewidth,
            fill_in_data,
            out,
            {100000: "$t_1"},
        )
    except Exception as e:
        logger.warning(
            f"Could not create plot {out.relative_to(out.parents[2])}, got exception {e}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("log_dir", type=str)
    args = parser.parse_args()
    log_dir = Path(args.log_dir)
    results = []
    with multiprocessing.Pool(20) as pool:
        for env_name, run_config in RUN_CONFIGS.items():
            env_file_name = env_name[:-3].lower().replace("-", "_")
            if not env_name.startswith("dmc"):
                env_file_name = "gym_" + env_file_name
            for out_filename, plot_config in PLOT_CONFIGS.items():
                for loss_type, loss_type_short in [
                    ("combined_loss", "combined"),
                    ("policy_loss", "policy"),
                    ("value_function_loss", "vf"),
                ]:
                    out_path = (
                        out_dir
                        / plot_config["out_dir"]
                        / loss_type
                        / f"{env_file_name}_{out_filename}_{loss_type_short}.pdf"
                    )
                    out_path.parent.mkdir(exist_ok=True, parents=True)
                    title_env_name = (
                        env_name[4:-3] if env_name.startswith("dmc_") else env_name[:-3]
                    )
                    keys = [
                        f"high_curvature_subspace_analysis/"
                        f"{analysis_run_id}/{plot_config['key']}/{loss_type}"
                        for analysis_run_id in run_config.get(
                            "analysis_run_ids", {"": "default"}
                        ).values()
                    ]
                    # Give default the lowest priority
                    keys.sort(key=lambda s: 1 if "default" in s else 0)
                    results.append(
                        pool.apply_async(
                            worker,
                            (
                                [
                                    Path(log_dir, env_name, experiment_dir)
                                    for experiment_dir in run_config[
                                        "log_dirs"
                                    ].values()
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
                                keys,
                                plot_config["smoothing_weight"],
                                True,
                                True,
                                plot_config.get("num_same_color_plots", 1),
                                plot_config.get("fontsize", 26),
                                plot_config.get("linewidth", 3),
                                plot_config.get("fill_in_data", {}),
                                out_path,
                            ),
                        )
                    )
        for res in tqdm(results):
            res.get()
