import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from matplotlib import pyplot as plt

from action_space_toolbox.util.tensorboard_logs import (
    create_event_accumulators,
    calculate_mean_std_sequence,
    read_scalar,
)


logger = logging.Logger(__name__)


def smooth(
    scalars: Sequence[float], weight: float
) -> np.ndarray:  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = []
    for val in scalars:
        smoothed_val = last * weight + (1 - weight) * val  # Calculate smoothed value
        smoothed.append(smoothed_val)  # Save it
        last = smoothed_val  # Anchor the last smoothed value

    return np.array(smoothed)


def create_multiline_plots(
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
    fill_in_data: Dict[int, float],
    out: Path,
) -> None:
    if legend is not None and len(keys) != len(legend):
        logger.warning(
            f"keys and legend do not have the same number of elements, keys: {len(keys)}, legend: {len(legend)}"
        )
    plt.rc("font", size=12)
    fig, ax = plt.subplots()
    ax.margins(x=0)
    if (log_path / "tensorboard").exists():
        run_dirs = [log_path]
    else:
        run_dirs = [d for d in log_path.iterdir() if d.is_dir() and d.name.isnumeric()]
    if len(run_dirs) > 0:
        tb_dirs = [run_dir / "tensorboard" for run_dir in run_dirs]
        event_accumulators = [ea for _, ea in create_event_accumulators(tb_dirs)]
        color = None  # To make PyLint happy
        linestyles = ["-", "--", "-.", ":"]
        for i, key in enumerate(keys):
            (
                steps,
                _,
                value_mean,
                value_std,
            ) = calculate_mean_std_sequence(event_accumulators, key)

            for s, v in fill_in_data.items():
                if s not in steps:
                    idx = np.argmax(steps > s)
                    steps = np.insert(steps, idx, s)
                    value_mean = np.insert(value_mean, idx, v)
                    value_std = np.insert(value_std, idx, 0.0)
            value_mean = smooth(value_mean, smoothing_weight)
            if value_std is not None:
                value_std = smooth(value_std, smoothing_weight)

            if xaxis_log:
                steps = 10**steps
                ax.xscale("log")
            if i % num_same_color_plots == 0:
                color = next(ax._get_lines.prop_cycler)["color"]
            ax.plot(
                steps,
                value_mean,
                marker=marker,
                markersize=2,
                color=color,
                linestyle=linestyles[i % num_same_color_plots],
            )
            if value_std is not None:
                ax.fill_between(
                    steps,
                    value_mean - value_std,
                    value_mean + value_std,
                    alpha=0.2,
                    label="_nolegend_",
                    color=color,
                )
    else:
        if (log_path / "tensorboard").exists():
            tb_dir = log_path / "tensorboard"
        else:
            tb_dir = log_path
        _, event_accumulator = create_event_accumulators([tb_dir])[0]
        key_indices = np.argwhere(
            [key in event_accumulator.Tags()["scalars"] for key in keys]
        )
        assert (
            key_indices.shape[0] > 0
        ), f"None of the keys {', '.join(keys)} is present in all tensorboard logs of {log_path}."
        key = keys[key_indices[0].item()]
        scalar = read_scalar(event_accumulator, key)
        scalar = list(scalar.items())
        scalar.sort(key=lambda x: x[0])

        steps = np.array([s[0] for s in scalar])
        if xaxis_log:
            steps = 10**steps
            ax.set_xscale("log")
        ax.plot(
            steps,
            smooth([s[1].value for s in scalar], smoothing_weight),
        )

    if not xaxis_log:
        ax.ticklabel_format(style="sci", axis="x", scilimits=(-4, 4), useMathText=True)
    ax.ticklabel_format(style="sci", axis="y", scilimits=(-4, 4), useMathText=True)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(xlimits)
    ax.set_ylim(ylimits)
    # To avoid cramming ticks labels too close together in the origin
    ax.tick_params(axis="x", pad=8)
    ax.tick_params(axis="y", pad=8)
    ax.xaxis.set_major_locator(plt.MaxNLocator(6))
    ax.yaxis.set_major_locator(plt.MaxNLocator(6))
    if legend is None:
        # The default legend entry consists of the parts that are the same in all keys
        keys_split = [key.split("/") for key in keys]
        key_indices_to_keep = [
            i
            for i in range(len(keys_split[0]))
            if sum(
                [
                    i < len(key_test) and keys_split[0][i] == key_test[i]
                    for key_test in keys_split[1:]
                ]
            )
            < len(keys_split) - 1
        ]
        legend = [
            "/".join([key_split[i] for i in key_indices_to_keep])
            for key_split in keys_split
        ]
    if not separate_legend:
        ax.legend(legend)
    fig.tight_layout(pad=0.1)
    out.parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(out.with_suffix(".pdf"))
    if legend is not None and separate_legend:
        legend_plt = ax.legend(
            legend, frameon=False, ncol=len(legend), bbox_to_anchor=(2.0, 2.0)
        )
        legend_fig = legend_plt.figure
        legend_fig.canvas.draw()
        bbox = legend_plt.get_window_extent().transformed(
            legend_fig.dpi_scale_trans.inverted()
        )
        legend_fig.savefig(out.parent / (out.name + "_legend.pdf"), bbox_inches=bbox)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("log_path", type=str)
    parser.add_argument("--keys", type=str, nargs="+", default=["rollout/ep_rew_mean"])
    parser.add_argument("--legend", type=str, nargs="+")
    parser.add_argument("--title", type=str, default="")
    parser.add_argument("--xlabel", type=str, default="Environment steps")
    parser.add_argument("--ylabel", type=str, default="Cumulative reward")
    parser.add_argument("--xmin", type=float)
    parser.add_argument("--xmax", type=float)
    parser.add_argument("--ymin", type=float)
    parser.add_argument("--ymax", type=float)
    parser.add_argument("--xaxis-log", action="store_true")
    parser.add_argument("--smoothing-weight", type=float, default=0.6)
    parser.add_argument("--separate-legend", action="store_true")
    parser.add_argument("--num-same-color-plots", type=int, default=1)
    parser.add_argument("--marker", default="d")
    parser.add_argument("--outname", type=str, default="graphs.pdf")
    args = parser.parse_args()

    out_dir = Path(__file__).parent.parent.parent / "out"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / args.outname

    create_multiline_plots(
        Path(args.log_path),
        args.legend,
        args.title,
        args.xlabel,
        args.ylabel,
        (args.xmin, args.xmax),
        (args.ymin, args.ymax),
        args.xaxis_log,
        args.keys,
        args.smoothing_weight,
        args.separate_legend,
        args.num_same_color_plots,
        args.marker,
        {},
        out,
    )
