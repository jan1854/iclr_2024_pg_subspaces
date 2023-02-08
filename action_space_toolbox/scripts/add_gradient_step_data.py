import functools
import itertools
import pickle
from argparse import ArgumentParser
from pathlib import Path

import gym
import hydra
import torch
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from action_space_toolbox.analysis.reward_surface_visualization.plotting import (
    plot_results,
)
from action_space_toolbox.analysis.reward_surface_visualization.reward_surface_visualization import (
    RewardSurfaceVisualization,
)

parser = ArgumentParser()
parser.add_argument("logdir", type=str)
parser.add_argument("--id", type=str, default="default")
parser.add_argument("--overwrite", action="store_true")
args = parser.parse_args()

log_dir = Path(args.logdir) / "training"

for env_dir in tqdm(log_dir.iterdir()):
    for day_dir in env_dir.iterdir():
        for experiment_dir in day_dir.iterdir():
            for i in itertools.count():
                if (experiment_dir / str(i)).exists():
                    run_dir = experiment_dir / str(i)
                else:
                    break
                analysis_dir = (
                    run_dir / "analyses" / "reward_surface_visualization" / "default"
                )
                if not (analysis_dir / "loss_surface" / "data").exists():
                    continue
                print(run_dir)

                summary_writer = SummaryWriter(str(run_dir / "tensorboard"))

                for loss_surface_data_path in (
                    analysis_dir / "loss_surface" / "data"
                ).iterdir():
                    agent_step = int(loss_surface_data_path.name[-14:-7])
                    plot_num = int(loss_surface_data_path.name[-6:-4])

                    with loss_surface_data_path.open("rb") as loss_surface_data_file:
                        loss_surface_data = pickle.load(loss_surface_data_file)

                    direction = loss_surface_data["directions"]

                    if (
                        "sampled_projected_optimizer_steps" in loss_surface_data
                        and "sampled_projected_sgd_steps" in loss_surface_data
                    ) and not args.overwrite:
                        continue

                    train_cfg = OmegaConf.load(run_dir / ".hydra" / "config.yaml")
                    agent_class = hydra.utils.get_class(
                        train_cfg.algorithm.algorithm._target_
                    )
                    device = "cpu"

                    env = gym.make(train_cfg.env, **train_cfg.env_args)

                    agent_checkpoint = (
                        run_dir
                        / "checkpoints"
                        / f"{train_cfg.algorithm.name}_{agent_step}_steps"
                    )
                    analysis = RewardSurfaceVisualization(
                        "default",
                        env_factory=functools.partial(
                            gym.make, train_cfg.env, **train_cfg.env_args
                        ),
                        agent_factory=functools.partial(
                            agent_class.load,
                            agent_checkpoint,
                            device=device,
                            tensorboard_log=None,
                        ),
                        run_dir=run_dir,
                        magnitude=loss_surface_data["magnitude"],
                        num_steps=agent_step,
                        num_plots=1,
                        num_processes=1,
                        grid_size=31,
                    )
                    (
                        optimizer_steps,
                        sgd_steps,
                    ) = analysis.sample_projected_update_trajectories(
                        [torch.tensor(d) for d in direction[0]],
                        [torch.tensor(d) for d in direction[1]],
                        10,
                    )

                    for concrete_analysis in analysis_dir.iterdir():
                        data_dir = concrete_analysis / "data"
                        found = False
                        for data_file in data_dir.iterdir():
                            if data_file.name.endswith(
                                f"{agent_step:07d}_{plot_num:02d}.pkl"
                            ):
                                found = True
                                with data_file.open("rb") as f:
                                    data_to_update = pickle.load(f)
                                    data_to_update[
                                        "sampled_projected_optimizer_steps"
                                    ] = optimizer_steps
                                    data_to_update[
                                        "sampled_projected_sgd_steps"
                                    ] = sgd_steps
                                with data_file.open("wb") as f:
                                    pickle.dump(data_to_update, f)
                        if not found:
                            raise RuntimeError()

                    logs = plot_results(
                        analysis_dir, agent_step, plot_num, overwrite=True
                    )
                    logs.log(summary_writer)
                    (run_dir / ".new_data").touch()
