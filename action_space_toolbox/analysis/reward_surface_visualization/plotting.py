import pickle
from pathlib import Path
from typing import Optional, Callable, Union

import PIL.Image
import numpy as np
import plotly.express
import plotly.graph_objects
from scipy.interpolate import RegularGridInterpolator

from action_space_toolbox.util.tensorboard_logs import TensorboardLogs


PLOT_NAME_TO_DESCR = {
    "reward_surface_undiscounted": "reward (undiscounted)",
    "reward_surface_discounted": "reward (discounted) surface",
    "policy_loss_surface": "policy loss",
    "negative_policy_loss_surface": "negative policy loss",
    "value_function_loss_surface": "value function loss",
    "negative_value_function_loss_surface": "negative value function loss",
    "loss_surface": "loss",
    "negative_loss_surface": "negative loss",
}


def plot_results(
    analysis_dir: Path,
    step: int,
    plot_num: int,
    overwrite: bool = False,
    plot_sgd_steps: bool = True,
    plot_true_gradient_steps: bool = True,
    max_gradient_trajectories: Optional[int] = None,
    max_steps_per_gradient_trajectory: Optional[int] = None,
) -> TensorboardLogs:
    logs = TensorboardLogs()

    for plot_name, plot_descr in PLOT_NAME_TO_DESCR.items():
        for results_path in (analysis_dir / plot_name / "data").glob(
            f"*{step:07d}_{plot_num:02d}*"
        ):
            plot_path = results_path.parents[1] / results_path.with_suffix("").name

            if not plot_path.exists() or overwrite:
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

                plot_surface(
                    results["magnitude"],
                    data,
                    results["env_name"],
                    plot_name,
                    results["env_step"],
                    results["plot_num"],
                    plot_descr,
                    results.get("gradient_direction"),
                    optimizer_steps,
                    sgd_steps,
                    optimizer_steps_true_grad,
                    sgd_steps_true_grad,
                    logs,
                    analysis_dir.name,
                    plot_path,
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
    gradient_direction: Optional[int],
    projected_optimizer_steps: Optional[np.ndarray],
    projected_sgd_steps: Optional[np.ndarray],
    projected_optimizer_steps_true_grad: Optional[np.ndarray],
    projected_sgd_steps_true_grad: Optional[np.ndarray],
    logs: TensorboardLogs,
    analysis_run_id: str,
    outpath: Path,
) -> None:
    outpath.parent.mkdir(parents=True, exist_ok=True)
    title = f"{env_name} | {descr} surface | magnitude: {magnitude}"

    coords = np.linspace(-magnitude, magnitude, num=results.shape[0])

    if gradient_direction is not None:
        yaxis_title = (
            "Gradient direction" if gradient_direction == 0 else "Random direction"
        )
        xaxis_title = (
            "Gradient direction" if gradient_direction == 1 else "Random direction"
        )
    else:
        yaxis_title = "Random direction 1"
        xaxis_title = "Random direction 2"
    fig = plotly.graph_objects.Figure(
        layout=plotly.graph_objects.Layout(
            margin=plotly.graph_objects.layout.Margin(l=20, r=20, t=50, b=25),
            scene={
                "aspectmode": "cube",
                "yaxis_title": yaxis_title,
                "xaxis_title": xaxis_title,
                "zaxis_title": descr,
            },
        )
    )

    # Plot surface
    fig.add_surface(
        z=results,
        x=coords,
        y=coords,
        colorscale="RdBu",
        reversescale=True,
    )

    z_range = abs(np.max(results) - np.min(results))
    # Add a black line at (0,0) to mark the current policy
    fig.add_scatter3d(
        x=[0.0, 0.0],
        y=[0.0, 0.0],
        z=[np.min(results) - 0.1 * z_range, np.max(results) + 0.1 * z_range],
        mode="lines",
        line_width=11,
        line_color="black",
        showlegend=False,
    )
    interpolator = RegularGridInterpolator((coords, coords), results, method="linear")

    if projected_optimizer_steps is not None:
        _plot_gradient_steps(
            fig,
            projected_optimizer_steps,
            interpolator,
            magnitude,
            z_range,
            "Optimizer trajectory",
            "black",
        )
    if projected_sgd_steps is not None:
        _plot_gradient_steps(
            fig,
            projected_sgd_steps,
            interpolator,
            magnitude,
            z_range,
            "SGD trajectory",
            "#FFFF00",
        )
    if projected_optimizer_steps_true_grad is not None:
        _plot_gradient_steps(
            fig,
            projected_optimizer_steps_true_grad,
            interpolator,
            magnitude,
            z_range,
            "True gradient optimizer trajectory",
            "#158463",
        )
    if projected_sgd_steps_true_grad is not None:
        _plot_gradient_steps(
            fig,
            projected_sgd_steps_true_grad,
            interpolator,
            magnitude,
            z_range,
            "True gradient SGD trajectory",
            "#FF6600",
        )

    if title is not None:
        fig.update_layout(title=title)
    # All the trajectories are disabled initially anyway, therefore we hide the legend for the png output
    fig.update_layout(showlegend=False)
    fig.write_image(outpath.with_suffix(".png"), scale=2)
    fig.update_layout(showlegend=True)
    fig.update_layout(legend=dict(x=1.06))
    fig.write_html(outpath.with_suffix(".html"))
    with PIL.Image.open(outpath.with_suffix(".png")) as im:
        # Make the image smaller so that it fits better in tensorboard
        im = im.resize((im.width // 2, im.height // 2), PIL.Image.Resampling.LANCZOS)
        logs.add_image(f"{plot_name}/{analysis_run_id}/{plot_nr}", im, env_step)


def _plot_gradient_steps(
    fig: plotly.graph_objects.Figure,
    projected_trajectories: np.ndarray,
    interpolator: Callable[[np.ndarray], np.ndarray],
    magnitude: float,
    z_range: float,
    name: str,
    color: str,
    opacity: float = 0.5,
) -> None:
    # Backward compatibility
    if projected_trajectories.ndim == 2:
        projected_trajectories = projected_trajectories[None]

    for i, grad_steps in enumerate(projected_trajectories):
        name_curr_trajectory = (
            f"{name} ({i + 1})" if len(projected_trajectories) > 1 else name
        )
        # Make sure that the gradient step does not point outside the grid (otherwise the interpolation will throw an
        # error). Scale the gradient step to reduce the length but keep the direction the same.
        visualization_steps = []
        grad_steps_start_end = []
        last_step = np.zeros(2)
        for step in grad_steps:
            if np.any(np.abs(step) > magnitude) and np.any(
                np.abs(last_step) > magnitude
            ):
                last_step = step
                continue
            else:
                step_rescaled = _rescale_if_out_of_bounds(step, magnitude)
                last_step_rescaled = _rescale_if_out_of_bounds(last_step, magnitude)
                grad_steps_start_end.append((last_step_rescaled, step_rescaled))
                last_step = step
        visualization_steps_per_gradient_step = 200 // len(grad_steps_start_end)
        for last_step, step in grad_steps_start_end:
            visualization_steps.append(
                np.linspace(
                    last_step,
                    step,
                    visualization_steps_per_gradient_step,
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
            np.mean(np.linalg.norm(grad_steps[1:] - grad_steps[:-1], axis=-1))
            > 0.005 * magnitude
        ):
            markers = projected_trajectories[
                np.all(np.abs(projected_trajectories) < magnitude, axis=-1), :
            ]
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
