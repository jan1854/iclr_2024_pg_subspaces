from argparse import ArgumentParser
from pathlib import Path
from typing import Optional

from torch.utils.tensorboard import SummaryWriter

from action_space_toolbox.analysis.reward_surface_visualization.plotting import (
    plot_results,
)


def create_reward_surface_plots(
    analysis_dir: Path,
    summary_writer: Optional[SummaryWriter],
    plot_sgd_steps: bool,
    plot_true_gradient_steps: bool,
    max_gradient_trajectories: int,
    max_steps_per_gradient_trajectory: Optional[int],
    disable_title: bool,
    outdir: Optional[Path],
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
            disable_title=disable_title,
            outdir=outdir,
        )
        if summary_writer is not None:
            logs.log(summary_writer)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("logdir", type=str)
    parser.add_argument("--id", type=str, default="default")
    parser.add_argument("--disable-sgd-steps", action="store_true")
    parser.add_argument("--disable-true-gradient-steps", action="store_true")
    parser.add_argument("--max-gradient-trajectories", type=int, default=1)
    parser.add_argument("--max-steps-per-gradient-trajectory", type=int)
    parser.add_argument("--disable-title", action="store_true")
    parser.add_argument("--outdir", type=str)
    args = parser.parse_args()

    logdir = Path(args.logdir)
    analysis_dir = logdir / "analyses" / "reward_surface_visualization" / args.id
    # Only add the generated images to the tensorboard logs if the plots are stored in the training logs
    if args.outdir is not None:
        outdir = Path(args.outdir)
        summary_writer = None
    else:
        outdir = None
        summary_writer = SummaryWriter(str(logdir / "tensorboard"))

    create_reward_surface_plots(
        analysis_dir,
        summary_writer,
        not args.disable_sgd_steps,
        not args.disable_true_gradient_steps,
        args.max_gradient_trajectories,
        args.max_steps_per_gradient_trajectory,
        args.disable_title,
        outdir,
    )
