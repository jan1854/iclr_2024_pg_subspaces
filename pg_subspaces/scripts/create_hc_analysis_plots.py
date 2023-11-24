import argparse
import logging
import multiprocessing
import traceback
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import yaml
from tqdm import tqdm

# Disable the loggers for the imported scripts (since these just spam too much)
logging.basicConfig(level=logging.CRITICAL)

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

PLOT_CONFIGS_SINGLE_RUN = {
    # "gradient_subspace_fraction_precise_vs_low_sample_hessian": {
    #     # "title": "Subspace frac., precise vs. low-sample Hessian",
    #     "out_dir": "gradient_subspace_fraction",
    #     "keys": [
    #         "gradient_subspace_fraction_001evs/estimated_gradient",
    #         "low_sample/gradient_subspace_fraction_001evs/estimated_gradient",
    #         "gradient_subspace_fraction_010evs/estimated_gradient",
    #         "low_sample/gradient_subspace_fraction_010evs/estimated_gradient",
    #         "gradient_subspace_fraction_100evs/estimated_gradient",
    #         "low_sample/gradient_subspace_fraction_100evs/estimated_gradient",
    #     ],
    #     "legend": [
    #         "True Hessian; 1 EV",
    #         "Estimated Hessian; 1 EV",
    #         "True Hessian; 10 EVs",
    #         "Estimated Hessian; 10 EVs",
    #         "True Hessian; 100 EVs",
    #         "Estimated Hessian; 100 EVs",
    #     ],
    #     "ylabel": "Gradient subspace fraction",
    #     "num_same_color_plots": 2,
    #     "ymin": 0.0,
    #     "ymax": 1.0,
    #     "smoothing_weight": 0.0,
    # },
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
        "xlabel": "Env. steps",
        "ylabel": "Gradient subspace frac.",
        "num_same_color_plots": 2,
        "ymin": 0.0,
        "ymax": 1.0,
        "smoothing_weight": 0.0,
    },
    # "true_gradient_subspace_fraction": {
    #     # "title": "True gradient subspace fraction",
    #     "out_dir": "gradient_subspace_fraction",
    #     "keys": [
    #         "gradient_subspace_fraction_001evs/true_gradient",
    #         "gradient_subspace_fraction_002evs/true_gradient",
    #         "gradient_subspace_fraction_005evs/true_gradient",
    #         "gradient_subspace_fraction_010evs/true_gradient",
    #         "gradient_subspace_fraction_020evs/true_gradient",
    #         "gradient_subspace_fraction_050evs/true_gradient",
    #         "gradient_subspace_fraction_100evs/true_gradient",
    #     ],
    #     "legend": ["1 EV", "2 EVs", "5 EVs", "10 EVs", "20 EVs", "50 EVs", "100 EVs"],
    #     "ylabel": "Gradient subspace fraction",
    #     "ymin": 0.0,
    #     "ymax": 1.0,
    #     "smoothing_weight": 0.0,
    # },
    # "estimated_gradient_subspace_fraction": {
    #     # "title": "Est. gradient subspace fraction",
    #     "out_dir": "gradient_subspace_fraction",
    #     "keys": [
    #         "gradient_subspace_fraction_001evs/estimated_gradient",
    #         "gradient_subspace_fraction_002evs/estimated_gradient",
    #         "gradient_subspace_fraction_005evs/estimated_gradient",
    #         "gradient_subspace_fraction_010evs/estimated_gradient",
    #         "gradient_subspace_fraction_020evs/estimated_gradient",
    #         "gradient_subspace_fraction_050evs/estimated_gradient",
    #         "gradient_subspace_fraction_100evs/estimated_gradient",
    #     ],
    #     "legend": ["1 EV", "2 EVs", "5 EVs", "10 EVs", "20 EVs", "50 EVs", "100 EVs"],
    #     "ylabel": "Gradient subspace fraction",
    #     "ymin": 0.0,
    #     "ymax": 1.0,
    #     "smoothing_weight": 0.0,
    # },
    # "subspace_overlap_0100000_precise_vs_low_sample_hessian": {
    #     # "title": "Subspace overlap, precise vs. low-sample Hessian, $t_1 = 10^{5}$",
    #     "out_dir": "subspace_overlap",
    #     "keys": [
    #         "overlaps_top001_checkpoint0100000",
    #         "low_sample/overlaps_top001_checkpoint0100000",
    #         "overlaps_top010_checkpoint0100000",
    #         "low_sample/overlaps_top010_checkpoint0100000",
    #         "overlaps_top100_checkpoint0100000",
    #         "low_sample/overlaps_top100_checkpoint0100000",
    #     ],
    #     "legend": [
    #         "True Hessian; 1 EV",
    #         "Estimated Hessian; 1 EV",
    #         "True Hessian; 10 EVs",
    #         "Estimated Hessian; 10 EVs",
    #         "True Hessian; 100 EVs",
    #         "Estimated Hessian; 100 EVs",
    #     ],
    #     "ylabel": "Subspace overlap",
    #     "num_same_color_plots": 2,
    #     "ymin": 0.0,
    #     "ymax": 1.0,
    #     "smoothing_weight": 0.0,
    #     "fill_in_data": {100000: 1.0},
    #     "x_annotations": [(100000, "$t_1$", False)],
    # },
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
        "x_annotations": [(10000, "$t_1$", False)],
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
        "x_annotations": [(100000, "$t_1$", False)],
    },
    # "subspace_overlap_0500000": {
    #     # "title": "Subspace overlap, $t_1 = 5 \cdot 10^{5}$",
    #     "out_dir": "subspace_overlap",
    #     "keys": [
    #         "overlaps_top001_checkpoint0500000",
    #         "overlaps_top010_checkpoint0500000",
    #         "overlaps_top100_checkpoint0500000",
    #     ],
    #     "legend": ["1 EV", "10 EVs", "100 EVs"],
    #     "xlabel": "$t_2$",
    #     "ylabel": "Subspace overlap",
    #     "ymin": 0.0,
    #     "ymax": 1.0,
    #     "smoothing_weight": 0.0,
    #     "fill_in_data": {500000: 1.0},
    #     "x_annotations": [(500000, "$t_1$", False)],
    # },
}

PLOT_CONFIGS_MULTIPLE_RUNS = {
    "gradient_subspace_fraction_true_010evs": {
        "out_dir": "gradient_subspace_fraction",
        "key": "gradient_subspace_fraction_010evs/true_gradient",
        "ylabel": "Gradient subspace fraction",
        "ymin": 0.0,
        "ymax": 1.0,
        "smoothing_weight": 0.0,
    },
    "gradient_subspace_fraction_true_100evs": {
        "out_dir": "gradient_subspace_fraction",
        "key": "gradient_subspace_fraction_100evs/true_gradient",
        "ylabel": "Gradient subspace fraction",
        "ymin": 0.0,
        "ymax": 1.0,
        "smoothing_weight": 0.0,
    },
    "subspace_overlap_010evs_checkpoint_0100000": {
        "out_dir": "subspace_overlap",
        "key": "overlaps_top100_checkpoint0100000",
        "ylabel": "Subspace overlap",
        "ymin": 0.0,
        "ymax": 1.0,
        "smoothing_weight": 0.0,
        "fill_in_data": {100000: 1.0},
        "x_annotations": [(100000, "$t_1$", False)],
    },
    "subspace_overlap_100evs_checkpoint_0100000": {
        "out_dir": "subspace_overlap",
        "key": "overlaps_top100_checkpoint0100000",
        "ylabel": "Subspace overlap",
        "ymin": 0.0,
        "ymax": 1.0,
        "smoothing_weight": 0.0,
        "fill_in_data": {100000: 1.0},
        "x_annotations": [(100000, "$t_1$", False)],
    },
}


def worker_single_run(
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
    only_complete_steps: bool,
    separate_legend: bool,
    num_same_color_plots: int,
    marker: Optional[str],
    fontsize: int,
    linewidth: float,
    fill_in_data: Dict[int, float],
    x_annotations: Sequence[Tuple[int, str, bool]],
    y_annotations: Sequence[Tuple[int, str, bool]],
    out: Path,
):
    from pg_subspaces.scripts.create_multiline_plots import create_multiline_plots

    try:
        create_multiline_plots(
            [log_path],
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
            marker,
            fontsize,
            linewidth,
            fill_in_data,
            x_annotations,
            y_annotations,
            out,
        )
    except Exception as e:
        logger.warning(
            f"Could not create plot {out.relative_to(out.parents[2])}, got exception {type(e).__name__}: {e}"
        )


def worker_multiple_runs(
    log_paths: Sequence[Path],
    legend: Optional[Sequence[str]],
    title: str,
    xlabel: str,
    ylabel: str,
    xlimits: Tuple[float, float],
    ylimits: Tuple[float, float],
    xaxis_log: bool,
    key: Sequence[str],
    smoothing_weight: float,
    only_complete_steps: bool,
    separate_legend: bool,
    num_same_color_plots: int,
    fontsize: int,
    linewidth: float,
    fill_in_data: Dict[int, float],
    x_annotations: Sequence[Tuple[int, str, bool]],
    y_annotations: Sequence[Tuple[int, str, bool]],
    out: Path,
):
    from pg_subspaces.scripts.create_plots import create_plots

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
            key,
            smoothing_weight,
            only_complete_steps,
            separate_legend,
            num_same_color_plots,
            fontsize,
            linewidth,
            fill_in_data,
            x_annotations,
            y_annotations,
            out,
        )
    except Exception as e:
        logger.warning(
            f"Could not create plot {out.relative_to(out.parents[2])}, got exception {e}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("log_dir", type=str)
    parser.add_argument("envs", type=str, nargs="*", default=["*"])
    parser.add_argument("--out", type=str)
    args = parser.parse_args()
    log_dir = Path(args.log_dir)
    if args.out is not None:
        out_dir = Path(args.out)
    else:
        out_dir = Path(__file__).parents[2] / "out"
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
            for algo_name, algo_log_dir in run_config["log_dirs"].items():
                curr_log_dir = log_dir / "training" / env_name / algo_log_dir
                env_file_name = env_name[:-3].lower().replace("-", "_")
                if not env_name.startswith("dmc"):
                    if env_name.startswith("Fetch"):
                        env_file_name = "gym-robotics_" + env_file_name
                    else:
                        env_file_name = "gym_" + env_file_name
                for out_filename, plot_config in PLOT_CONFIGS_SINGLE_RUN.items():
                    for loss_type, loss_type_short in [
                        ("combined_loss", "combined"),
                        ("policy_loss", "policy"),
                        ("value_function_loss", "vf"),
                    ]:
                        experiment_out_dir = None
                        for (
                            experiment_name,
                            experiment_config,
                        ) in experiment_configs.items():
                            if (
                                experiment_out_dir is None or experiment_name == "main"
                            ) and algo_name in experiment_config:
                                experiment_out_dir = experiment_name
                        out_path = (
                            out_dir
                            / experiment_out_dir
                            / plot_config["out_dir"]
                            / loss_type
                            / f"{algo_name}_{env_file_name}_{out_filename}_{loss_type_short}.pdf"
                        )
                        if "analysis_run_ids" in run_config and isinstance(
                            run_config["analysis_run_ids"], dict
                        ):
                            analysis_run_id = run_config["analysis_run_ids"].get(
                                algo_name, "default"
                            )
                        else:
                            analysis_run_id = run_config.get(
                                "anlysis_run_ids", "default"
                            )
                        keys = [
                            f"high_curvature_subspace_analysis/{analysis_run_id}/{key}/{loss_type}"
                            for key in plot_config["keys"]
                        ]
                        if env_name.startswith("dmc_"):
                            title_env_name = env_name[len("dmc_") : -3]
                        elif env_name.startswith("gym-robotics_"):
                            title_env_name = env_name[len("gym-robotics_") : -3]
                        else:
                            title_env_name = env_name[:-3]
                        title = (
                            f"{algo_name.upper()} - {title_env_name} - {plot_config.get('title')}"
                            if plot_config.get("title") is not None
                            else None
                        )
                        results.append(
                            pool.apply_async(
                                worker_single_run,
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
                                    plot_config.get("smoothing_weight", 0.0),
                                    True,
                                    True,
                                    plot_config.get("num_same_color_plots", 1),
                                    plot_config.get("marker"),
                                    plot_config.get("fontsize", 26),
                                    plot_config.get("linewidth", 2.5),
                                    plot_config.get("fill_in_data", {}),
                                    plot_config.get("x_annotations", {}),
                                    plot_config.get("y_annotations", {}),
                                    out_path,
                                ),
                            )
                        )

        for experiment_name, experiment_config in experiment_configs.items():
            for env_name, run_config in run_configs.items():
                if env_name not in args.envs and "*" not in args.envs:
                    continue
                env_file_name = env_name[:-3].lower().replace("-", "_")
                if not env_name.startswith("dmc"):
                    env_file_name = "gym_" + env_file_name
                log_paths = [
                    log_dir / "training" / env_name / run_config["log_dirs"][algo_name]
                    if algo_name in run_config["log_dirs"]
                    else None
                    for algo_name in experiment_config.keys()
                ]
                analysis_run_ids = [
                    a_id
                    for a_id in run_config.get("analysis_run_ids", {})
                    if a_id != "default"
                ] + ["default"]
                for out_filename, plot_config in PLOT_CONFIGS_MULTIPLE_RUNS.items():
                    for loss_type, loss_type_short in [
                        ("combined_loss", "combined"),
                        ("policy_loss", "policy"),
                        ("value_function_loss", "vf"),
                    ]:
                        out_path = (
                            out_dir
                            / experiment_name
                            / plot_config["out_dir"]
                            / loss_type
                            / f"{env_file_name}_{out_filename}_{loss_type_short}.pdf"
                        )
                        keys = [
                            f"high_curvature_subspace_analysis/{a_id}/{plot_config['key']}/{loss_type}"
                            for a_id in analysis_run_ids
                        ] + [
                            f"subspace_overlaps_analysis/{a_id}/{plot_config['key']}/{loss_type}"
                            for a_id in analysis_run_ids
                        ]
                        results.append(
                            pool.apply_async(
                                worker_multiple_runs,
                                (
                                    log_paths,
                                    list(experiment_config.values()),
                                    None,
                                    plot_config.get("xlabel", "Environment steps"),
                                    plot_config["ylabel"],
                                    (run_config.get("xmin", 0), run_config.get("xmax")),
                                    (plot_config.get("ymin"), plot_config.get("ymax")),
                                    False,
                                    keys,
                                    plot_config.get("smoothing_weight", 0.0),
                                    True,
                                    True,
                                    plot_config.get("num_same_color_plots", 1),
                                    plot_config.get("fontsize", 22),
                                    plot_config.get("linewidth", 2.5),
                                    plot_config.get("fill_in_data", {}),
                                    plot_config.get("x_annotations", ()),
                                    plot_config.get("y_annotations", ()),
                                    out_path,
                                ),
                            )
                        )

        for res in tqdm(results):
            res.get()
