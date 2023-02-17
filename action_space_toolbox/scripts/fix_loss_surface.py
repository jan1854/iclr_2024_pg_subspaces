import functools
import itertools
import pickle
from argparse import ArgumentParser
from pathlib import Path

import gym
import hydra
import numpy as np
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
from action_space_toolbox.util.agent_spec import AgentSpec

parser = ArgumentParser()
parser.add_argument("logdir", type=str)
parser.add_argument("--id", type=str, default="default")
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

                    directions = loss_surface_data["directions"]

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
                    agent_spec = AgentSpec(
                        agent_checkpoint,
                        device,
                        agent_kwargs={"tensorboard_log": None},
                    )
                    analysis = RewardSurfaceVisualization(
                        "default",
                        env_factory=functools.partial(
                            gym.make, train_cfg.env, **train_cfg.env_args
                        ),
                        agent_spec=agent_spec,
                        run_dir=run_dir,
                        magnitude=loss_surface_data["magnitude"],
                        num_steps=agent_step,
                        num_plots=1,
                        num_processes=1,
                        grid_size=31,
                    )
                    grid_size = loss_surface_data["data"].shape[0]
                    magnitude = loss_surface_data["magnitude"]
                    agent = agent_spec.create_agent(env)
                    agent_weights = [p.data.detach() for p in agent.policy.parameters()]
                    weights_offsets = [[None] * grid_size for _ in range(grid_size)]
                    coords = np.linspace(-magnitude, magnitude, num=grid_size)

                    for offset1_idx, offset1_scalar in enumerate(coords):
                        weights_curr_offset1 = [
                            a_weight + off * offset1_scalar
                            for a_weight, off in zip(agent_weights, directions[0])
                        ]
                        for offset2_idx, offset2_scalar in enumerate(coords):
                            weights_curr_offsets = [
                                a_weight + off * offset2_scalar
                                for a_weight, off in zip(
                                    weights_curr_offset1, directions[1]
                                )
                            ]
                            weights_offsets[offset1_idx][
                                offset2_idx
                            ] = weights_curr_offsets

                    weights_offsets_flat = [
                        item for sublist in weights_offsets for item in sublist
                    ]

                    losses, policy_ratios = analysis.loss_surface_analysis(
                        weights_offsets,
                        analysis.env_factory,
                        analysis.agent_spec,
                        200000,
                    )

                    for data, concrete_analysis in zip(
                        [-losses, losses, None, None],
                        [
                            analysis_dir / "negative_loss_surface",
                            analysis_dir / "loss_surface",
                            analysis_dir / "reward_surface_discounted",
                            analysis_dir / "reward_surface_undiscounted",
                        ],
                    ):
                        data_dir = concrete_analysis / "data"
                        found = False
                        for data_file in data_dir.iterdir():
                            if data_file.name.endswith(
                                f"{agent_step:07d}_{plot_num:02d}.pkl"
                            ):
                                found = True
                                with data_file.open("rb") as f:
                                    data_to_update = pickle.load(f)
                                    if data is not None:
                                        data_to_update["data"] = data
                                    data_to_update["policy_ratio"] = policy_ratios
                                with data_file.open("wb") as f:
                                    pickle.dump(data_to_update, f)
                        if not found:
                            raise RuntimeError()

                    logs = plot_results(
                        analysis_dir, agent_step, plot_num, overwrite=True
                    )
                    logs.log(summary_writer)
