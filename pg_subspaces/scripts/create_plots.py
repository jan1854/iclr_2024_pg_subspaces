import argparse
import logging
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from matplotlib import pyplot as plt

from pg_subspaces.metrics.read_metrics_cached import read_metrics_cached


logger = logging.Logger(__name__)


def calculate_axis_limits(
    curr_min: float, curr_max: float, val: float
) -> Tuple[float, float]:
    val_min = val + (curr_max - val) * 0.1
    val_max = val - (val - curr_min) * 0.1
    return min(val_min, curr_min), max(val_max, curr_max)


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


def create_plots(
    log_paths: Sequence[Optional[Path]],
    legend: Sequence[str],
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
    fontsize: int,
    linewidth: float,
    fill_in_data: Dict[int, float],
    x_annotations: Sequence[Tuple[float, str, bool]],  # position, text, legend?
    y_annotations: Sequence[Tuple[float, str, bool]],  # position, text, legend?
    out: Path,
) -> None:
    log_paths_filtered = [l for l in log_paths if l is not None]
    legend = list(legend)
    if len(legend) > 0 and len(log_paths_filtered) != len(legend):
        logger.warning(
            f'log_paths (without "skip"s) and legend do not have the same number of elements, '
            f"log_paths: {len(log_paths_filtered)}, legend: {len(legend)}"
        )
    plt.rc("font", size=fontsize)
    plt.rc("axes", linewidth=linewidth)
    plt.rc("lines", linewidth=linewidth * 1.5)
    fig, ax = plt.subplots()
    try:
        ax.margins(x=0)
        color = None  # To make PyLint happy
        linestyles = ["-", "--", "-.", ":"]
        for i, log_path in enumerate(log_paths):
            if i % num_same_color_plots == 0:
                color = next(ax._get_lines.prop_cycler)["color"]
            if log_path is not None:
                metrics = read_metrics_cached(log_path, keys)
                min_last_step = min([m[0][-1] for m in metrics])
                max_last_step = max([m[0][-1] for m in metrics])
                if min_last_step != max_last_step:
                    logger.warning(
                        f"Found different last step ({min_last_step} vs. {max_last_step}), "
                        f"using {'minimum' if only_complete_steps else 'maximum'} value."
                    )
                    if only_complete_steps:
                        metrics = [list(metric) for metric in metrics]
                        for metric in metrics:
                            metric[0] = np.array(
                                [m for m in metric[0] if m <= min_last_step]
                            )
                            metric[1] = metric[1][: len(metric[0])]
                        metrics = [tuple(metric) for metric in metrics]

                steps = metrics[0][0]
                value_mean = np.array(
                    [
                        np.mean([m[1][i] for m in metrics if i < len(m[1])])
                        for i in range(len(steps))
                    ]
                )
                value_std = np.array(
                    [
                        np.std([m[1][i] for m in metrics if i < len(m[1])])
                        for i in range(len(steps))
                    ]
                )

                for s, v in fill_in_data.items():
                    if s not in steps:
                        idx = np.argmax(steps > s)
                        steps = np.insert(steps, idx, s)
                        value_mean = np.insert(value_mean, idx, v)
                        value_std = np.insert(value_std, idx, 0.0)

                value_mean = smooth(value_mean, smoothing_weight)
                value_std = smooth(value_std, smoothing_weight)

                if xaxis_log:
                    steps = 10**steps
                    ax.xscale("log")
                ax.plot(
                    steps,
                    value_mean,
                    color=color,
                    linestyle=linestyles[i % num_same_color_plots],
                    zorder=i + 2,
                )
                ax.fill_between(
                    steps,
                    value_mean - value_std,
                    value_mean + value_std,
                    alpha=0.2,
                    label="_nolegend_",
                    color=color,
                    zorder=i + 52,
                )

        if not xaxis_log:
            ax.ticklabel_format(
                style="sci", axis="x", scilimits=(-4, 4), useMathText=True
            )
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

        for pos, annotation, ann_legend in x_annotations:
            y_low, y_high = ax.get_ylim()
            ax.axvline(
                x=pos,
                color="gray",
                linestyle="--",
                label=annotation if ann_legend else "_nolegend_",
                zorder=0,
            )
            if ann_legend:
                legend.append(annotation)
            else:
                # We want the text to be in front of the graphs but the white bounding box should just be in front of
                # the horizontal line (but behind the graphs), this is achieved by the following hack that first adds an
                # empty text box with white bounding box and then adds the text (without bounding box).
                plt.text(
                    pos,
                    (y_high + y_low) / 2,
                    annotation,
                    color="white",
                    verticalalignment="center",
                    horizontalalignment="center",
                    bbox=dict(
                        facecolor="white", edgecolor="none", boxstyle="square,pad=0.05"
                    ),
                    zorder=1,
                )
                plt.text(
                    pos,
                    (y_high + y_low) / 2,
                    annotation,
                    verticalalignment="center",
                    horizontalalignment="center",
                    color="gray",
                    zorder=100,
                )
            x_low, x_high = ax.get_xlim()
            x_low, x_high = calculate_axis_limits(x_low, x_high, pos)
            ax.set_xlim(x_low, x_high)

        for pos, annotation, ann_legend in y_annotations:
            x_low, x_high = ax.get_xlim()
            ax.axhline(
                y=pos,
                color="gray",
                linestyle="--",
                label=annotation if ann_legend else "_nolegend_",
                zorder=0,
            )
            if ann_legend:
                legend.append(annotation)
            else:
                # We want the text to be in front of the graphs but the white bounding box should just be in front of
                # the vertical line (but behind the graphs), this is achieved by the following hack that first adds an
                # empty text box with white bounding box and then adds the text (without bounding box).
                plt.text(
                    (x_high + x_low) / 2,
                    pos,
                    annotation,
                    color="white",
                    verticalalignment="center",
                    horizontalalignment="center",
                    bbox=dict(
                        facecolor="white", edgecolor="none", boxstyle="square,pad=0.05"
                    ),
                    zorder=1,
                )
                plt.text(
                    (x_high + x_low) / 2,
                    pos,
                    annotation,
                    verticalalignment="center",
                    horizontalalignment="center",
                    color="gray",
                    zorder=100,
                )
            y_low, y_high = ax.get_ylim()
            y_low, y_high = calculate_axis_limits(y_low, y_high, pos)
            ax.set_ylim(y_low, y_high)

        if legend is not None and not separate_legend:
            ax.legend(legend, loc="lower right")
        fig.tight_layout(pad=0.1)
        out.parent.mkdir(exist_ok=True, parents=True)
        fig.savefig(out.with_suffix(".pdf"))
        if legend is not None and separate_legend:
            legend_plt = ax.legend(
                legend,
                frameon=False,
                ncol=len(legend),
                bbox_to_anchor=(2.0, 2.0),
            )
            legend_fig = legend_plt.figure
            legend_fig.canvas.draw()
            bbox = legend_plt.get_window_extent().transformed(
                legend_fig.dpi_scale_trans.inverted()
            )
            # For some reason this is necessary since otherwise the legend pdf is empty (don't ask why :D)
            bbox.x0 -= 0.1
            bbox.x1 += 0.1
            legend_fig.savefig(
                out.parent / (out.stem + "_legend.pdf"), bbox_inches=bbox
            )
    finally:
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("log_paths", type=str, nargs="+")
    parser.add_argument("--key", type=str, nargs="+", default=["rollout/ep_rew_mean"])
    parser.add_argument("--legend", type=str, nargs="+", default=())
    parser.add_argument("--title", type=str, default="")
    parser.add_argument("--xlabel", type=str, default="Environment steps")
    parser.add_argument("--ylabel", type=str, default="")
    parser.add_argument("--xmin", type=float)
    parser.add_argument("--xmax", type=float)
    parser.add_argument("--ymin", type=float)
    parser.add_argument("--ymax", type=float)
    parser.add_argument("--xaxis-log", action="store_true")
    parser.add_argument("--smoothing-weight", type=float, default=0.6)
    parser.add_argument("--use-incomplete-steps", action="store_true")
    parser.add_argument("--separate-legend", action="store_true")
    parser.add_argument("--num-same-color-plots", type=int, default=1)
    parser.add_argument("--fontsize", type=int, default=12)
    parser.add_argument("--linewidth", type=float, default=1.5)
    parser.add_argument("--x-annotations", type=str, nargs="+", default=())
    parser.add_argument("--y-annotations", type=str, nargs="+", default=())
    parser.add_argument("--outname", type=str, default="graphs.pdf")
    args = parser.parse_args()

    out_dir = Path(__file__).parent.parent.parent / "out"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / args.outname

    x_annotations_parsed = []
    y_annotations_parsed = []
    for x_ann in args.x_annotations:
        pos, ann, legend = x_ann.split(":")
        x_annotations_parsed.append((float(pos), ann, legend.lower() == "true"))
    for y_ann in args.y_annotations:
        pos, ann, legend = y_ann.split(":")
        y_annotations_parsed.append((float(pos), ann, legend.lower() == "true"))

    create_plots(
        [
            Path(log_path) if not log_path == "skip" else None
            for log_path in args.log_paths
        ],
        args.legend,
        args.title,
        args.xlabel,
        args.ylabel,
        (args.xmin, args.xmax),
        (args.ymin, args.ymax),
        args.xaxis_log,
        args.key,
        args.smoothing_weight,
        not args.use_incomplete_steps,
        args.separate_legend,
        args.num_same_color_plots,
        args.fontsize,
        args.linewidth,
        {},
        x_annotations_parsed,
        y_annotations_parsed,
        out,
    )
