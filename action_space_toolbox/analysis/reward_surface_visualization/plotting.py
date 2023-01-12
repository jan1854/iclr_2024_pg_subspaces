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
    "loss_surface": "loss surface",
    "negative_loss_surface": "negative loss surface",
}


def plot_all_results(analysis_dir: Path, overwrite=False) -> TensorboardLogs:
    logs = TensorboardLogs()

    for plot_name, plot_title in PLOT_NAME_TO_TITLE_DESCR.items():
        for results_path in (analysis_dir / plot_name / "data").iterdir():
            plot_path = results_path.parents[1] / results_path.with_suffix("").name

            if not plot_path.exists() or overwrite:
                with results_path.open("rb") as results_file:
                    results = pickle.load(results_file)
                data = results["data"]

                plot_surface(
                    results["magnitude"],
                    data,
                    results["env_name"],
                    plot_name,
                    results["env_step"],
                    results["plot_num"],
                    plot_title,
                    results.get("sampled_projected_gradient_steps", []),
                    logs,
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
    projected_gradient_steps: np.ndarray,
    logs: TensorboardLogs,
    outpath: Path,
) -> None:
    outpath.parent.mkdir(parents=True, exist_ok=True)
    title = f"{env_name} | {title_descr} | magnitude: {magnitude}"

    coords = np.linspace(-magnitude, magnitude, num=results.shape[0])

    fig = plotly.graph_objects.Figure(
        layout=plotly.graph_objects.Layout(
            margin=plotly.graph_objects.layout.Margin(l=20, r=20, t=50, b=25)
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
    fig.add_scatter3d(
        x=[0.0, 0.0],
        y=[0.0, 0.0],
        z=[np.min(results) - 0.1 * Z_range, np.max(results) + 0.1 * Z_range],
        mode="lines",
        line_width=8,
        line_color="black",
        showlegend=False,
    )
    interpolator = RegularGridInterpolator((coords, coords), results, method="linear")

    for grad_step in projected_gradient_steps:
        # Make sure that the gradient step does not point outside the grid (otherwise the interpolation will throw an
        # error).
        grad_step = np.clip(grad_step, -magnitude + 1e-6, magnitude - 1e-6)
        visualization_steps = np.linspace(np.zeros(2), grad_step, 100)
        # TODO: Check out the "Setting Angle Reference" example at https://plotly.com/python/marker-style
        fig.add_scatter3d(
            x=visualization_steps[:, 1],
            y=visualization_steps[:, 0],
            z=interpolator(visualization_steps) + 0.001 * Z_range,
            mode="lines",
            line_width=6,
            line_color="black",
            showlegend=False,
            opacity=0.2,
        )

    if title is not None:
        fig.update_layout(title=title)

    fig.write_image(outpath.with_suffix(".png"), scale=5)
    fig.write_html(outpath.with_suffix(".html"))
    with Image.open(outpath.with_suffix(".png")) as im:
        # Make the image smaller so that it fits better in tensorboard
        im = im.resize((im.width // 5, im.height // 5), PIL.Image.Resampling.LANCZOS)
        logs.add_image(f"{plot_name}/{plot_nr}", im, env_step)
