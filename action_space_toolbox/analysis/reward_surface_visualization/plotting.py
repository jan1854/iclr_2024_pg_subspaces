import pickle
from pathlib import Path

import PIL.Image
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt, cm
from mpl_toolkits.mplot3d import Axes3D

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
            plot_path = results_path.parents[1] / results_path.with_suffix(".png").name

            if not plot_path.exists() or overwrite:
                with results_path.open("rb") as results_file:
                    results = pickle.load(results_file)
                data = results["data"]
                coords = (
                    np.linspace(-1.0, 1.0, num=data.shape[0]) * results["magnitude"]
                )

                plot_surface(
                    coords,
                    data,
                    results["env_name"],
                    plot_name,
                    results["env_step"],
                    results["plot_num"],
                    plot_title,
                    logs,
                    plot_path,
                )
    return logs


def plot_surface(
    coords: np.ndarray,
    results: np.ndarray,
    env_name: str,
    plot_name: str,
    env_step: int,
    plot_nr: int,
    title_descr: str,
    logs: TensorboardLogs,
    outpath: Path,
) -> None:
    x_coords, y_coords = np.meshgrid(coords, coords)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    title = f"{env_name} {title_descr}"

    fig = plt.figure()
    ax = Axes3D(fig)

    if title is not None:
        fig.suptitle(title)

    # Plot surface
    surf = ax.plot_surface(
        x_coords,
        y_coords,
        results,
        cmap=cm.coolwarm,
        linewidth=0,
        antialiased=False,
        zorder=5,
    )

    # Plot center line above surface
    Z_range = abs(np.max(results) - np.min(results))
    zline_above = np.linspace(
        results[len(results) // 2][len(results[0]) // 2],
        np.max(results) + (Z_range * 0.1),
        4,
    )
    xline_above = 0 * zline_above
    yline_above = 0 * zline_above
    ax.plot3D(xline_above, yline_above, zline_above, "black", zorder=10)

    # Plot center line below surface
    zline_below = np.linspace(
        results[len(results) // 2][len(results[0]) // 2],
        np.min(results) - (Z_range * 0.1),
        4,
    )
    xline_below = 0 * zline_below
    yline_below = 0 * zline_below
    ax.plot3D(xline_below, yline_below, zline_below, "black", zorder=0)

    fig.colorbar(surf, shrink=0.5, aspect=5, pad=0.05)

    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)

    with Image.open(outpath) as im:
        # Make the image smaller so that it fits better in tensorboard
        im = im.resize((im.width // 2, im.height // 2), PIL.Image.Resampling.LANCZOS)
        logs.add_image(f"{plot_name}/{plot_nr}", im, env_step)
