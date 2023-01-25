import argparse
from pathlib import Path
from typing import Sequence, Tuple

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
    legend: Sequence[str],
    title: str,
    xlabel: str,
    ylabel: str,
    xlimits: Tuple[float, float],
    ylimits: Tuple[float, float],
    key: str,
    smoothing_weight: float,
    out: Path,
) -> None:
    for log_path, name in zip(log_paths, legend):
        run_dirs = [d for d in log_path.iterdir() if d.is_dir() and d.name.isnumeric()]
        if len(run_dirs) > 0:
            tb_dirs = [run_dir / "tensorboard" for run_dir in run_dirs]
            event_accumulators = [ea for _, ea in create_event_accumulators(tb_dirs)]
            (
                steps,
                _,
                value_mean,
                value_std,
            ) = calculate_mean_std_sequence(event_accumulators, key)

            value_mean = smooth(value_mean, smoothing_weight)
            value_std = smooth(value_std, smoothing_weight)

            plt.plot(steps, value_mean, label=name)
            plt.fill_between(
                steps, value_mean - value_std, value_mean + value_std, alpha=0.2
            )
        else:
            tb_dir = log_path / "tensorboard"
            _, event_accumulator = create_event_accumulators([tb_dir])[0]
            scalar = read_scalar(event_accumulator, key)

            plt.plot(list(scalar.keys()), list(scalar.values()), label=name)
    plt.ticklabel_format(axis="x", style="sci", scilimits=(0, 4))
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim(xlimits)
    plt.ylim(ylimits)
    plt.legend(loc="lower right")
    plt.savefig(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("log_paths", type=str, nargs="+")
    parser.add_argument("--key", type=str, default="rollout/ep_rew_mean")
    parser.add_argument("--legend", type=str, nargs="+")
    parser.add_argument("--title", type=str, default="")
    parser.add_argument("--xlabel", type=str, default="Environment steps")
    parser.add_argument("--ylabel", type=str, default="Cumulative reward")
    parser.add_argument("--xmin", type=float)
    parser.add_argument("--xmax", type=float)
    parser.add_argument("--ymin", type=float)
    parser.add_argument("--ymax", type=float)
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
        args.key,
        args.smoothing_weight,
        out,
    )
