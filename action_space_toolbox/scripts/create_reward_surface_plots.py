from argparse import ArgumentParser
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

from action_space_toolbox.analysis.reward_surface_visualization.plotting import (
    plot_all_results,
)


def create_reward_surface_plots(
    analysis_dir: Path,
    summary_writer: SummaryWriter,
) -> None:
    logs = plot_all_results(analysis_dir, overwrite=True)
    logs.log(summary_writer)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("logdir", type=str)
    args = parser.parse_args()

    logdir = Path(args.logdir)
    analysis_dir = logdir / "analyses" / "reward_surface_visualization"
    summary_writer = SummaryWriter(str(logdir / "tensorboard"))
    create_reward_surface_plots(analysis_dir, summary_writer)
