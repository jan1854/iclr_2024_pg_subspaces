import argparse
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
from matplotlib import pyplot as plt

from action_space_toolbox.util.tensorboard_logs import (
    create_event_accumulators,
    calculate_mean_std_sequence,
    read_scalar,
)


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
    log_paths: Sequence[Path],
    legend: Optional[Sequence[str]],
    title: str,
    xlabel: str,
    ylabel: str,
    xlimits: Tuple[float, float],
    ylimits: Tuple[float, float],
    xaxis_log: bool,
    keys: List[str],
    smoothing_weight: float,
    out: Path,
) -> None:
    ax = plt.gca()
    for log_path in log_paths:
        run_dirs = [d for d in log_path.iterdir() if d.is_dir() and d.name.isnumeric()]
        if len(run_dirs) > 0:
            tb_dirs = [run_dir / "tensorboard" for run_dir in run_dirs]
            event_accumulators = [ea for _, ea in create_event_accumulators(tb_dirs)]
            key_indices = np.argwhere(
                [
                    np.all([key in ea.Tags()["scalars"] for ea in event_accumulators])
                    for key in keys
                ]
            )
            assert (
                key_indices.shape[0] > 0
            ), f"None of the keys {', '.join(keys)} is present in all tensorboard logs of {log_path}."
            key = keys[key_indices[0].item()]
            (
                steps,
                _,
                value_mean,
                value_std,
            ) = calculate_mean_std_sequence(event_accumulators, key)

            value_mean = smooth(value_mean, smoothing_weight)
            value_std = smooth(value_std, smoothing_weight)

            if xaxis_log:
                steps = 10**steps
                plt.xscale("log")
            color = next(ax._get_lines.prop_cycler)["color"]
            plt.plot(steps, value_mean, color=color)
            plt.fill_between(
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
                plt.xscale("log")
            plt.plot(
                steps,
                smooth([s[1].value for s in scalar], smoothing_weight),
            )
    if not xaxis_log:
        plt.ticklabel_format(axis="x", style="sci", scilimits=(0, 4))
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim(xlimits)
    plt.ylim(ylimits)
    if legend is not None:
        plt.legend(legend, loc="lower right")
    plt.savefig(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("log_paths", type=str, nargs="+")
    parser.add_argument("--key", type=str, nargs="+", default=["rollout/ep_rew_mean"])
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
    parser.add_argument("--outname", type=str, default="graphs.pdf")
    args = parser.parse_args()

    out_dir = Path(__file__).parent.parent.parent / "out"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / args.outname

    create_plots(
        [Path(log_path) for log_path in args.log_paths],
        args.legend,
        args.title,
        args.xlabel,
        args.ylabel,
        (args.xmin, args.xmax),
        (args.ymin, args.ymax),
        args.xaxis_log,
        args.key,
        args.smoothing_weight,
        out,
    )
