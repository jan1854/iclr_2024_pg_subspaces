from argparse import ArgumentParser
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

from action_space_toolbox.analysis.reward_surface_visualization.plotting import (
    plot_results,
)


def create_reward_surface_plots(
    analysis_dir: Path,
    summary_writer: SummaryWriter,
) -> None:
    for data_file in (analysis_dir / "loss_surface" / "data").iterdir():
        step = int(data_file.name[-14:-7])
        plot_num = int(data_file.name[-6:-4])
        logs = plot_results(analysis_dir, step, plot_num, overwrite=True)
        logs.log(summary_writer)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("logdir", type=str)
    parser.add_argument("--id", type=str, default="default")
    args = parser.parse_args()

    logdir = Path(args.logdir)
    analysis_dir = logdir / "analyses" / "reward_surface_visualization" / args.id
    summary_writer = SummaryWriter(str(logdir / "tensorboard"))
    create_reward_surface_plots(analysis_dir, summary_writer)
