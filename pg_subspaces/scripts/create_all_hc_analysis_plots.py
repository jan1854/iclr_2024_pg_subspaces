import argparse
import logging
import multiprocessing
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

# Disable the loggers for the imported scripts (since these just spam too much)
logging.basicConfig(level=logging.CRITICAL)

from tqdm import tqdm

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

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
    "Pendulum_PC-v1": {"log_dirs": {"ppo": "2022-11-18/17-24-43/0"}, "xmax": 300_000},
    "Pendulum_VC-v1": {"log_dirs": {"ppo": "2022-11-21/19-55-23/0"}, "xmax": 300_000},
    "Reacher_PC-v2": {"log_dirs": {"ppo": "2022-11-14/13-57-50/0"}, "xmax": 1_000_000},
    "Reacher_VC-v2": {"log_dirs": {"ppo": "2023-01-13/16-01-01/0"}, "xmax": 1_000_000},
    "Walker2d_TC-v3": {
        "log_dirs": {"ppo": "2023-07-14/23-14-41", "sac": "2023-09-23/23-20-09"},
        "xmax": 2_000_000,
        "analysis_run_ids": {"ppo": "repeat_low_sample", "sac": "default"},
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
    "gradient_subspace_fraction_precise_vs_low_sample_hessian": {
        # "title": "Subspace frac., precise vs. low-sample Hessian",
        "out_dir": "gradient_subspace_fraction",
        "keys": [
            "gradient_subspace_fraction_001evs/estimated_gradient",
            "low_sample/gradient_subspace_fraction_001evs/estimated_gradient",
            "gradient_subspace_fraction_010evs/estimated_gradient",
            "low_sample/gradient_subspace_fraction_010evs/estimated_gradient",
            "gradient_subspace_fraction_100evs/estimated_gradient",
            "low_sample/gradient_subspace_fraction_100evs/estimated_gradient",
        ],
        "legend": [
            "True Hessian; 1 EV",
            "Estimated Hessian; 1 EV",
            "True Hessian; 10 EVs",
            "Estimated Hessian; 10 EVs",
            "True Hessian; 100 EVs",
            "Estimated Hessian; 100 EVs",
        ],
        "ylabel": "Gradient subspace fraction",
        "num_same_color_plots": 2,
        "ymin": 0.0,
        "ymax": 1.0,
        "smoothing_weight": 0.0,
    },
    "gradient_subspace_fraction_estimated_vs_true": {
        # "title": "Subspace frac., estimated vs. true grad.",
        "out_dir": "gradient_subspace_fraction",
        "keys": [
            "gradient_subspace_fraction_001evs/true_gradient",
            "gradient_subspace_fraction_001evs/estimated_gradient",
            "gradient_subspace_fraction_010evs/true_gradient",
            "gradient_subspace_fraction_010evs/estimated_gradient",
            "gradient_subspace_fraction_100evs/true_gradient",
            "gradient_subspace_fraction_100evs/estimated_gradient",
        ],
        "legend": [
            "True grad.; 1 EV",
            "Estimated grad.; 1 EV",
            "True grad.; 10 EVs",
            "Estimated grad.; 10 EVs",
            "True grad.; 100 EVs",
            "Estimated grad.; 100 EVs",
        ],
        "ylabel": "Gradient subspace fraction",
        "num_same_color_plots": 2,
        "ymin": 0.0,
        "ymax": 1.0,
        "smoothing_weight": 0.0,
    },
    "true_gradient_subspace_fraction": {
        # "title": "True gradient subspace fraction",
        "out_dir": "gradient_subspace_fraction",
        "keys": [
            "gradient_subspace_fraction_001evs/true_gradient",
            "gradient_subspace_fraction_002evs/true_gradient",
            "gradient_subspace_fraction_005evs/true_gradient",
            "gradient_subspace_fraction_010evs/true_gradient",
            "gradient_subspace_fraction_020evs/true_gradient",
            "gradient_subspace_fraction_050evs/true_gradient",
            "gradient_subspace_fraction_100evs/true_gradient",
        ],
        "legend": ["1 EV", "2 EVs", "5 EVs", "10 EVs", "20 EVs", "50 EVs", "100 EVs"],
        "ylabel": "Gradient subspace fraction",
        "ymin": 0.0,
        "ymax": 1.0,
        "smoothing_weight": 0.0,
    },
    "estimated_gradient_subspace_fraction": {
        # "title": "Est. gradient subspace fraction",
        "out_dir": "gradient_subspace_fraction",
        "keys": [
            "gradient_subspace_fraction_001evs/estimated_gradient",
            "gradient_subspace_fraction_002evs/estimated_gradient",
            "gradient_subspace_fraction_005evs/estimated_gradient",
            "gradient_subspace_fraction_010evs/estimated_gradient",
            "gradient_subspace_fraction_020evs/estimated_gradient",
            "gradient_subspace_fraction_050evs/estimated_gradient",
            "gradient_subspace_fraction_100evs/estimated_gradient",
        ],
        "legend": ["1 EV", "2 EVs", "5 EVs", "10 EVs", "20 EVs", "50 EVs", "100 EVs"],
        "ylabel": "Gradient subspace fraction",
        "ymin": 0.0,
        "ymax": 1.0,
        "smoothing_weight": 0.0,
    },
    "subspace_overlap_0100000_precise_vs_low_sample_hessian": {
        # "title": "Subspace overlap, precise vs. low-sample Hessian, $t_1 = 10^{5}$",
        "out_dir": "subspace_overlap",
        "keys": [
            "overlaps_top001_checkpoint0100000",
            "low_sample/overlaps_top001_checkpoint0100000",
            "overlaps_top010_checkpoint0100000",
            "low_sample/overlaps_top010_checkpoint0100000",
            "overlaps_top100_checkpoint0100000",
            "low_sample/overlaps_top100_checkpoint0100000",
        ],
        "legend": [
            "True Hessian; 1 EV",
            "Estimated Hessian; 1 EV",
            "True Hessian; 10 EVs",
            "Estimated Hessian; 10 EVs",
            "True Hessian; 100 EVs",
            "Estimated Hessian; 100 EVs",
        ],
        "ylabel": "Subspace overlap",
        "num_same_color_plots": 2,
        "ymin": 0.0,
        "ymax": 1.0,
        "smoothing_weight": 0.0,
        "fill_in_data": {100000: 1.0},
    },
    "subspace_overlap_0010000": {
        # "title": "Subspace overlap, $t_1 = 10^{4}$",
        "out_dir": "subspace_overlap",
        "keys": [
            "overlaps_top001_checkpoint0010000",
            "overlaps_top010_checkpoint0010000",
            "overlaps_top100_checkpoint0010000",
        ],
        "legend": ["1 EV", "10 EVs", "100 EVs"],
        "xlabel": "$t_2$",
        "ylabel": "Subspace overlap",
        "ymin": 0.0,
        "ymax": 1.0,
        "smoothing_weight": 0.0,
        "fill_in_data": {10000: 1.0},
        "annotations": {10000: "$t_1$"},
    },
    "subspace_overlap_0100000": {
        # "title": "Subspace overlap, $t_1 = 10^{5}$",
        "out_dir": "subspace_overlap",
        "keys": [
            "overlaps_top001_checkpoint0100000",
            "overlaps_top010_checkpoint0100000",
            "overlaps_top100_checkpoint0100000",
        ],
        "legend": ["1 EV", "10 EVs", "100 EVs"],
        "xlabel": "$t_2$",
        "ylabel": "Subspace overlap",
        "ymin": 0.0,
        "ymax": 1.0,
        "smoothing_weight": 0.0,
        "fill_in_data": {100000: 1.0},
        "annotations": {100000: "$t_1$"},
    },
    "subspace_overlap_0500000": {
        # "title": "Subspace overlap, $t_1 = 5 \cdot 10^{5}$",
        "out_dir": "subspace_overlap",
        "keys": [
            "overlaps_top001_checkpoint0500000",
            "overlaps_top010_checkpoint0500000",
            "overlaps_top100_checkpoint0500000",
        ],
        "legend": ["1 EV", "10 EVs", "100 EVs"],
        "xlabel": "$t_2$",
        "ylabel": "Subspace overlap",
        "ymin": 0.0,
        "ymax": 1.0,
        "smoothing_weight": 0.0,
        "fill_in_data": {500000: 1.0},
        "annotations": {500000: "$t_1$"},
    },
}


def worker(
    log_path: Path,
    legend: Optional[Sequence[str]],
    title: str,
    xlabel: str,
    ylabel: str,
    xlimits: Tuple[float, float],
    ylimits: Tuple[float, float],
    xaxis_log: bool,
    keys: Sequence[str],
    smoothing_weight: float,
    separate_legend: bool,
    num_same_color_plots: int,
    marker: Optional[str],
    fontsize: int,
    linewidth: float,
    fill_in_data: Dict[int, float],
    out: Path,
    annotations: Dict[int, str],
):
    from scripts.create_multiline_plots import create_multiline_plots

    try:
        create_multiline_plots(
            log_path,
            legend,
            title,
            xlabel,
            ylabel,
            xlimits,
            ylimits,
            xaxis_log,
            keys,
            smoothing_weight,
            separate_legend,
            num_same_color_plots,
            marker,
            fontsize,
            linewidth,
            fill_in_data,
            out,
            annotations,
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
            for algo_name, algo_log_dir in run_config["log_dirs"].items():
                curr_log_dir = log_dir / env_name / algo_log_dir
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
                            / f"{algo_name}_{env_file_name}_{out_filename}_{loss_type_short}.pdf"
                        )
                        if "analysis_run_ids" in run_config and isinstance(
                            run_config["analysis_run_ids"], dict
                        ):
                            analysis_run_id = run_config["analysis_run_ids"][algo_name]
                        else:
                            analysis_run_id = run_config.get(
                                "anlysis_run_ids", "default"
                            )
                        keys = [
                            f"high_curvature_subspace_analysis/{analysis_run_id}/{key}/{loss_type}"
                            for key in plot_config["keys"]
                        ]
                        title_env_name = (
                            env_name[4:-3]
                            if env_name.startswith("dmc_")
                            else env_name[:-3]
                        )
                        title = (
                            f"{algo_name.upper()} - {title_env_name} - {plot_config.get('title')}"
                            if plot_config.get("title") is not None
                            else None
                        )
                        results.append(
                            pool.apply_async(
                                worker,
                                (
                                    curr_log_dir,
                                    plot_config["legend"],
                                    title,
                                    plot_config.get("xlabel", "Environment steps"),
                                    plot_config["ylabel"],
                                    (run_config.get("xmin", 0), run_config.get("xmax")),
                                    (plot_config.get("ymin"), plot_config.get("ymax")),
                                    False,
                                    keys,
                                    0.3,
                                    True,
                                    plot_config.get("num_same_color_plots", 1),
                                    plot_config.get("marker"),
                                    plot_config.get("fontsize", 22),
                                    plot_config.get("linewidth", 2.5),
                                    plot_config.get("fill_in_data", {}),
                                    out_path,
                                    plot_config.get("annotations", {}),
                                ),
                            )
                        )
        for res in tqdm(results):
            res.get()