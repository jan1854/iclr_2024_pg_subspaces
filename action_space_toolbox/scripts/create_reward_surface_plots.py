from argparse import ArgumentParser
from pathlib import Path
from typing import Optional

from torch.utils.tensorboard import SummaryWriter

from action_space_toolbox.analysis.reward_surface_visualization.plotting import (
    plot_results,
)


def create_reward_surface_plots(
    analysis_dir: Path,
    summary_writer: SummaryWriter,
    plot_sgd_steps: bool,
    plot_true_gradient_steps: bool,
    max_gradient_trajectories: int,
    max_steps_per_gradient_trajectory: Optional[int],
) -> None:
    for data_file in (analysis_dir / "loss_surface" / "data").iterdir():
        step = int(data_file.name[-14:-7])
        plot_num = int(data_file.name[-6:-4])
        logs = plot_results(
            analysis_dir,
            step,
            plot_num,
            overwrite=True,
            plot_sgd_steps=plot_sgd_steps,
            plot_true_gradient_steps=plot_true_gradient_steps,
            max_gradient_trajectories=max_gradient_trajectories,
            max_steps_per_gradient_trajectory=max_steps_per_gradient_trajectory,
        )
        logs.log(summary_writer)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("logdir", type=str)
    parser.add_argument("--id", type=str, default="default")
    parser.add_argument("--plot-sgd-steps", action="store_true")
    parser.add_argument("--plot-true-gradient-steps", action="store_true")
    parser.add_argument("--max-gradient-trajectories", type=int, default=1)
    parser.add_argument("--max-steps-per-gradient-trajectory", type=int)
    args = parser.parse_args()

    logdir = Path(args.logdir)
    analysis_dir = logdir / "analyses" / "reward_surface_visualization" / args.id
    summary_writer = SummaryWriter(str(logdir / "tensorboard"))
    create_reward_surface_plots(
        analysis_dir,
        summary_writer,
        args.plot_sgd_steps,
        args.plot_true_gradient_steps,
        args.max_gradient_trajectories,
        args.max_steps_per_gradient_trajectory,
    )
