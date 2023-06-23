import pickle
from pathlib import Path
from typing import Callable, Literal, Optional, Sequence, Tuple, Union

import PIL.Image
import numpy as np
import plotly.express
import plotly.graph_objects
import skimage.measure
from scipy.interpolate import RegularGridInterpolator

from action_space_toolbox.util.tensorboard_logs import TensorboardLogs


PLOT_NAME_TO_DESCR = {
    "reward_surface_undiscounted": "reward (undis.)",
    "reward_surface_discounted": "cum. reward",
    "policy_loss_surface": "policy loss",
    "negative_policy_loss_surface": "neg. policy loss",
    "value_function_loss_surface": "neg. function loss",
    "negative_value_function_loss_surface": "neg. value loss",
    "loss_surface": "loss",
    "negative_loss_surface": "neg. loss",
}

PLOT_MARGINS_WITH_TITLE = {"l": 20, "r": 20, "t": 50, "b": 25}
PLOT_MARGINS_WITHOUT_TITLE = {"l": 0, "r": 0, "t": 0, "b": 0, "pad": 0}


def plot_results(
    analysis_dir: Path,
    step: int,
    plot_num: int,
    overwrite: bool = False,
    plot_sgd_steps: bool = True,
    plot_true_gradient_steps: bool = True,
    max_gradient_trajectories: Optional[int] = None,
    max_steps_per_gradient_trajectory: Optional[int] = None,
    disable_title: bool = False,
    outdir_override: Optional[Path] = None,
) -> TensorboardLogs:
    logs = TensorboardLogs(
        f"reward_surface_visualization/{analysis_dir.name}",
        f"reward_surface_visualization_step_plots/{analysis_dir.name}",
    )

    for plot_name, plot_descr in PLOT_NAME_TO_DESCR.items():
        for results_path in (analysis_dir / plot_name / "data").glob(
            f"*{step:07d}_{plot_num:02d}*"
        ):
            if outdir_override is not None:
                outdir = outdir_override
            else:
                outdir = results_path.parents[1]

            outpath = outdir / results_path.with_suffix("").name

            if not outpath.exists() or overwrite:
                with results_path.open("rb") as results_file:
                    results = pickle.load(results_file)
                data = results["data"]

                if "sampled_projected_optimizer_steps" in results:
                    optimizer_steps = results["sampled_projected_optimizer_steps"]
                    optimizer_steps = optimizer_steps[
                        :max_gradient_trajectories, :max_steps_per_gradient_trajectory
                    ]
                else:
                    optimizer_steps = None
                if plot_sgd_steps and "sampled_projected_sgd_steps" in results:
                    sgd_steps = results["sampled_projected_sgd_steps"]
                    sgd_steps = sgd_steps[
                        :max_gradient_trajectories, :max_steps_per_gradient_trajectory
                    ]
                else:
                    sgd_steps = None
                if (
                    plot_true_gradient_steps
                    and "sampled_projected_optimizer_steps_true_gradient" in results
                ):
                    optimizer_steps_true_grad = results[
                        "sampled_projected_optimizer_steps_true_gradient"
                    ]
                    optimizer_steps_true_grad = optimizer_steps_true_grad[
                        :, :max_steps_per_gradient_trajectory
                    ]
                else:
                    optimizer_steps_true_grad = None
                if (
                    plot_sgd_steps
                    and plot_true_gradient_steps
                    and "sampled_projected_sgd_steps_true_gradient" in results
                ):
                    sgd_steps_true_grad = results[
                        "sampled_projected_sgd_steps_true_gradient"
                    ]
                    sgd_steps_true_grad = sgd_steps_true_grad[
                        :, :max_steps_per_gradient_trajectory
                    ]
                else:
                    sgd_steps_true_grad = None

                if "direction_types" in results:
                    direction_types = results["direction_types"]
                elif "gradient_direction" in results:
                    direction_types = ("grad", "rand")
                else:
                    direction_types = ("rand", "rand")

                plot_surface(
                    results["magnitude"],
                    data,
                    results["env_name"],
                    plot_name,
                    results["env_step"],
                    results["plot_num"],
                    plot_descr,
                    direction_types,
                    optimizer_steps,
                    sgd_steps,
                    optimizer_steps_true_grad,
                    sgd_steps_true_grad,
                    results.get("policy_ratio"),
                    disable_title,
                    logs,
                    outpath,
                )

    # Plot the policy ratios (the data is in every plot directory, so just take the first one from PLOT_NAME_TO_DESCR)
    data_dir_name = next(iter(PLOT_NAME_TO_DESCR.keys()))
    results_path = next(
        (analysis_dir / data_dir_name / "data").glob(f"*{step:07d}_{plot_num:02d}*")
    )
    with results_path.open("rb") as results_file:
        results = pickle.load(results_file)

    if "direction_types" in results:
        direction_types = results["direction_types"]
    elif "gradient_direction" in results:
        direction_types = ("grad", "rand")
    else:
        direction_types = ("rand", "rand")

    if "policy_ratio" in results:
        plot_surface(
            results["magnitude"],
            results["policy_ratio"],
            results["env_name"],
            "policy_ratio",
            results["env_step"],
            results["plot_num"],
            "policy ratio",
            direction_types,
            None,
            None,
            None,
            None,
            None,
            disable_title,
            logs,
            analysis_dir / "policy_ratios" / f"policy_ratios_{step:07d}",
        )
    return logs


def plot_surface(
    magnitude: float,
    results: np.ndarray,
    env_name: str,
    plot_name: str,
    env_step: int,
    plot_nr: int,
    descr: str,
    direction_types: Tuple[
        Literal["rand", "grad", "hess_ev"], Literal["rand", "grad", "hess_ev"]
    ],
    projected_optimizer_steps: Optional[np.ndarray],
    projected_sgd_steps: Optional[np.ndarray],
    projected_optimizer_steps_true_grad: Optional[np.ndarray],
    projected_sgd_steps_true_grad: Optional[np.ndarray],
    policy_ratios: Optional[np.ndarray],
    disable_title: bool,
    logs: TensorboardLogs,
    outpath: Path,
) -> None:
    outpath.parent.mkdir(parents=True, exist_ok=True)

    coords = np.linspace(-magnitude, magnitude, num=results.shape[0])

    axis_titles = []
    for dir_type in direction_types:
        if dir_type == "rand":
            axis_titles.append("random dir.")
        elif dir_type == "grad":
            axis_titles.append("gradient dir.")
        else:
            # TODO: This axis should also include the number of the ev (top-k ev)
            axis_titles.append("Hessian ev.")
    margins = PLOT_MARGINS_WITHOUT_TITLE if disable_title else PLOT_MARGINS_WITH_TITLE
    axis_label_fontsize = 26
    ticks_fontsize = 14
    fig = plotly.graph_objects.Figure(
        layout=plotly.graph_objects.Layout(
            margin=margins,
            scene={
                "aspectmode": "cube",
                "xaxis": {
                    "title": axis_titles[1],
                    "title_font": {"size": axis_label_fontsize},
                    "tickfont": {"size": ticks_fontsize},
                    "range": [-magnitude * 1.15, magnitude * 1.15],
                },
                "yaxis": {
                    "title": axis_titles[0],
                    "title_font": {"size": axis_label_fontsize},
                    "tickfont": {"size": ticks_fontsize},
                    "range": [-magnitude * 1.15, magnitude * 1.15],
                },
                "zaxis": {
                    "title": descr,
                    "title_font": {"size": axis_label_fontsize},
                    "tickfont": {"size": ticks_fontsize},
                },
            },
            scene_camera={
                "eye": {"x": 1.25, "y": 1.25, "z": 1.25},
                "center": {"z": -0.15},
            },
            font={"size": 16},
        )
    )

    # Plot surface
    fig.add_surface(
        z=results,
        x=coords,
        y=coords,
        colorscale="RdBu",
        reversescale=True,
        showscale=True,
        colorbar={"x": 0.9},
    )

    z_range = abs(np.max(results) - np.min(results))
    # Add a black line at (0,0) to mark the current policy
    fig.add_scatter3d(
        x=[0.0, 0.0],
        y=[0.0, 0.0],
        z=[np.min(results) - 0.1 * z_range, np.max(results) + 0.1 * z_range],
        mode="lines",
        line_width=6,
        line_color="black",
        showlegend=False,
    )
    interpolator = RegularGridInterpolator((coords, coords), results, method="linear")

    if projected_optimizer_steps is not None:
        _plot_curves(
            fig,
            _prepend_zeros(projected_optimizer_steps),
            interpolator,
            magnitude,
            z_range,
            "Optimizer trajectory",
            "black",
        )
    if projected_sgd_steps is not None:
        _plot_curves(
            fig,
            _prepend_zeros(projected_sgd_steps),
            interpolator,
            magnitude,
            z_range,
            "SGD trajectory",
            "#FFFF00",
        )
    if projected_optimizer_steps_true_grad is not None:
        _plot_curves(
            fig,
            _prepend_zeros(projected_optimizer_steps_true_grad),
            interpolator,
            magnitude,
            z_range,
            "True gradient optimizer trajectory",
            "#158463",
        )
    if projected_sgd_steps_true_grad is not None:
        _plot_curves(
            fig,
            _prepend_zeros(projected_sgd_steps_true_grad),
            interpolator,
            magnitude,
            z_range,
            "True gradient SGD trajectory",
            "#9723C2",
        )

    if policy_ratios is not None:
        clip_contour_lower, clip_contour_upper = _get_clipping_contours(
            magnitude, policy_ratios, 0.2
        )  # TODO: Clipping ratio should not be hardcoded
        _plot_curves(
            fig,
            clip_contour_lower,
            interpolator,
            magnitude,
            z_range,
            "Clipping range lower",
            "#595959",
            add_markers=False,
            single_legend_entry=True,
        )
        _plot_curves(
            fig,
            clip_contour_upper,
            interpolator,
            magnitude,
            z_range,
            "Clipping range upper",
            "#595959",
            add_markers=False,
            single_legend_entry=True,
        )

    if not disable_title:
        title = f"{env_name} | {descr} | magnitude: {magnitude}"
        fig.update_layout(title=title)

    # All the trajectories are disabled initially anyway, therefore we hide the legend for the png output
    fig.update_layout(showlegend=False)
    # Use height=490, width=576 to crop white space
    fig.write_image(outpath.with_suffix(".png"), scale=2)
    fig.update_layout(showlegend=True)
    fig.update_layout(legend=dict(x=1.06))
    fig.write_html(outpath.with_suffix(".html"))
    with PIL.Image.open(outpath.with_suffix(".png")) as im:
        # Make the image smaller so that it fits better in tensorboard
        im = im.resize((im.width // 2, im.height // 2), PIL.Image.Resampling.LANCZOS)
        logs.add_image(f"{plot_name}/{plot_nr}", im, env_step)


def _prepend_zeros(trajectories: np.ndarray) -> np.ndarray:
    # Backward compatibility
    if trajectories.ndim == 2:
        trajectories = trajectories[None]
    return np.concatenate(
        (np.zeros((trajectories.shape[0], 1, trajectories.shape[2])), trajectories),
        axis=1,
    )


def _get_clipping_contours(
    magnitude: float, policy_ratios: np.ndarray, clipping_range: float
):
    clip_contours_lower = skimage.measure.find_contours(
        policy_ratios, 1 - clipping_range
    )
    clip_contours_upper = skimage.measure.find_contours(
        policy_ratios, 1 + clipping_range
    )
    # Normalize the coordinates from [0, grid_size - 1] to [-magnitude, magnitude]
    clip_contours_lower = [
        (2 * contour / (policy_ratios.shape[0] - 1) - 1.0) * magnitude
        for contour in clip_contours_lower
    ]
    clip_contours_upper = [
        (2 * contour / (policy_ratios.shape[0] - 1) - 1.0) * magnitude
        for contour in clip_contours_upper
    ]
    return clip_contours_lower, clip_contours_upper


def _plot_curves(
    fig: plotly.graph_objects.Figure,
    curves: Union[np.ndarray, Sequence[np.ndarray]],
    interpolator: Callable[[np.ndarray], np.ndarray],
    magnitude: float,
    z_range: float,
    name: str,
    color: str,
    opacity: float = 0.5,
    add_markers: bool = True,
    single_legend_entry: bool = False,
) -> None:
    for i, curve in enumerate(curves):
        name_curr_trajectory = (
            f"{name} ({i + 1})" if len(curves) > 1 and not single_legend_entry else name
        )
        # Make sure that the curve segment does not point outside the grid (otherwise the interpolation will throw an
        # error). Scale the curve segment to reduce the length but keep the direction the same.
        visualization_steps = []
        curve_start_end = []
        last_step = curve[0]
        for step in curve[1:]:
            if np.any(np.abs(step) > magnitude) and np.any(
                np.abs(last_step) > magnitude
            ):
                last_step = step
            else:
                step_rescaled = _rescale_if_out_of_bounds(step, magnitude)
                last_step_rescaled = _rescale_if_out_of_bounds(last_step, magnitude)
                curve_start_end.append((last_step_rescaled, step_rescaled))
                last_step = step
        if len(curve_start_end) == 0:
            continue
        visualization_steps_per_curve_segment = 200 // len(curve_start_end)
        for last_step, step in curve_start_end:
            visualization_steps.append(
                np.linspace(
                    last_step,
                    step,
                    visualization_steps_per_curve_segment,
                )
            )
        visualization_steps = np.stack(visualization_steps)

        legend_entry_created = False
        for segment in visualization_steps:
            # TODO: Check out the "Setting Angle Reference" example at https://plotly.com/python/marker-style
            z_values_segment = interpolator(segment) + 0.01 * z_range
            fig.add_scatter3d(
                x=segment[:, 1],
                y=segment[:, 0],
                z=z_values_segment,
                mode="lines",
                line_width=8,
                line_color=color,
                name=name_curr_trajectory,
                showlegend=not legend_entry_created,
                legendgroup=name_curr_trajectory,
                opacity=opacity,
                visible="legendonly",
            )
            if not legend_entry_created:
                legend_entry_created = True
        # Only show the additional markers if the plot is zoomed in enough (the average distance between markers is
        # larger than some threshold) to avoid clutter
        if (
            add_markers
            and np.mean(np.linalg.norm(curve[1:] - curve[:-1], axis=-1))
            > 0.005 * magnitude
        ):
            markers = curve[np.all(np.abs(curve) < magnitude, axis=-1), :]
            assert np.all(np.abs(markers) < magnitude)
            z_values_markers = interpolator(markers) + 0.01 * z_range
            fig.add_scatter3d(
                x=markers[:, 1],
                y=markers[:, 0],
                z=z_values_markers,
                mode="markers",
                marker_size=2,
                marker_color=color,
                showlegend=False,
                name=name_curr_trajectory,
                legendgroup=name_curr_trajectory,
                opacity=opacity,
                visible="legendonly",
            )


def _rescale_if_out_of_bounds(arr: np.array, bounds: Union[int, np.array]) -> np.array:
    rescale_factors = bounds / (np.abs(arr) + 1e-8)
    if np.any(rescale_factors < 1.0):
        return arr * np.min(rescale_factors)
    else:
        return arr
