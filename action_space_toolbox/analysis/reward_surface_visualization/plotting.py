import pickle
from pathlib import Path

import PIL.Image
import numpy as np
import plotly.graph_objects
from PIL import Image
from scipy.interpolate import RegularGridInterpolator

from action_space_toolbox.util.tensorboard_logs import TensorboardLogs


PLOT_NAME_TO_TITLE_DESCR = {
    "reward_surface_undiscounted": "reward surface (undiscounted)",
    "reward_surface_discounted": "reward surface (discounted)",
    "policy_loss_surface": "policy loss surface",
    "negative_policy_loss_surface": "negative policy loss surface",
    "value_function_loss_surface": "value function loss surface",
    "negative_value_function_loss_surface": "negative value function loss surface",
    "loss_surface": "loss surface",
    "negative_loss_surface": "negative loss surface",
}


def plot_results(
    analysis_dir: Path,
    step: int,
    plot_num: int,
    overwrite: bool = False,
    plot_sgd_steps: bool = False,
) -> TensorboardLogs:
    logs = TensorboardLogs()

    for plot_name, plot_title in PLOT_NAME_TO_TITLE_DESCR.items():
        for results_path in (analysis_dir / plot_name / "data").glob(
            f"*{step:07d}_{plot_num:02d}*"
        ):
            plot_path = results_path.parents[1] / results_path.with_suffix("").name

            if not plot_path.exists() or overwrite:
                with results_path.open("rb") as results_file:
                    results = pickle.load(results_file)
                data = results["data"]

                sgd_steps = (
                    results.get("sampled_projected_sgd_steps", [])
                    if plot_sgd_steps
                    else []
                )
                plot_surface(
                    results["magnitude"],
                    data,
                    results["env_name"],
                    plot_name,
                    results["env_step"],
                    results["plot_num"],
                    plot_title,
                    results.get("sampled_projected_optimizer_steps", []),
                    sgd_steps,
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
    title_descr: str,
    projected_optimizer_steps: np.ndarray,
    projected_sgd_steps: np.ndarray,
    logs: TensorboardLogs,
    analysis_run_id: str,
    outpath: Path,
) -> None:
    outpath.parent.mkdir(parents=True, exist_ok=True)
    title = f"{env_name} | {title_descr} | magnitude: {magnitude}"

    coords = np.linspace(-magnitude, magnitude, num=results.shape[0])

    fig = plotly.graph_objects.Figure(
        layout=plotly.graph_objects.Layout(
            margin=plotly.graph_objects.layout.Margin(l=20, r=20, t=50, b=25),
            scene={"aspectmode": "cube"},
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

    Z_range = abs(np.max(results) - np.min(results))
    # Add a black line at (0,0) to mark the current policy
    fig.add_scatter3d(
        x=[0.0, 0.0],
        y=[0.0, 0.0],
        z=[np.min(results) - 0.1 * Z_range, np.max(results) + 0.1 * Z_range],
        mode="lines",
        line_width=10,
        line_color="black",
        showlegend=False,
    )
    interpolator = RegularGridInterpolator((coords, coords), results, method="linear")

    for grad_step in projected_sgd_steps:
        visualization_steps = np.linspace(np.zeros(2), grad_step, 200)
        fig.add_scatter3d(
            x=visualization_steps[:, 1],
            y=visualization_steps[:, 0],
            z=interpolator(visualization_steps) + 0.01 * Z_range,
            mode="lines",
            line_width=8,
            line_color="blue",
            showlegend=False,
            opacity=0.2,
        )

    for grad_step in projected_optimizer_steps:
        # Make sure that the gradient step does not point outside the grid (otherwise the interpolation will throw an
        # error). Scale the gradient step to reduce the length but keep the direction the same.
        if np.max(np.abs(grad_step)) > magnitude:
            grad_step = grad_step * magnitude / np.max(np.abs(grad_step)) - 1e-8
        visualization_steps = np.linspace(np.zeros(2), grad_step, 200)
        # TODO: Check out the "Setting Angle Reference" example at https://plotly.com/python/marker-style
        fig.add_scatter3d(
            x=visualization_steps[:, 1],
            y=visualization_steps[:, 0],
            z=interpolator(visualization_steps) + 0.01 * Z_range,
            mode="lines",
            line_width=8,
            line_color="black",
            showlegend=False,
            opacity=0.5,
        )

    if title is not None:
        fig.update_layout(title=title)

    fig.write_image(outpath.with_suffix(".png"), scale=2)
    fig.write_html(outpath.with_suffix(".html"))
    with Image.open(outpath.with_suffix(".png")) as im:
        # Make the image smaller so that it fits better in tensorboard
        im = im.resize((im.width // 2, im.height // 2), PIL.Image.Resampling.LANCZOS)
        logs.add_image(f"{plot_name}/{analysis_run_id}/{plot_nr}", im, env_step)
