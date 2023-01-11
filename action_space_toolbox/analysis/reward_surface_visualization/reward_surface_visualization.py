import functools
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence, Union

import gym
import numpy as np
import stable_baselines3.common.base_class
import stable_baselines3.common.buffers
import stable_baselines3.common.vec_env
import torch
from tqdm import tqdm

from action_space_toolbox.analysis.analysis import Analysis
from action_space_toolbox.analysis.reward_surface_visualization.plotting import (
    plot_all_results,
)
from action_space_toolbox.util.get_episode_length import get_episode_length
from action_space_toolbox.util.sb3_training import fill_rollout_buffer, ppo_loss
from action_space_toolbox.util.tensorboard_logs import TensorboardLogs


# To avoid too many open files problems when passing tensors between processes (see
# https://github.com/pytorch/pytorch/issues/11201#issuecomment-421146936)
torch.multiprocessing.set_sharing_strategy("file_system")
# To avoid the warning "Forking a process while a parallel region is active is potentially unsafe."
torch.set_num_threads(1)

logger = logging.getLogger(__name__)


@dataclass
class AnalysisResult:
    reward_undiscounted: float
    reward_discounted: float
    ppo_loss: float


class RewardSurfaceVisualization(Analysis):
    def __init__(
        self,
        env_factory: Callable[[], gym.Env],
        agent_factory: Callable[
            [Union[gym.Env, stable_baselines3.common.vec_env.VecEnv]],
            stable_baselines3.ppo.PPO,
        ],
        run_dir: Path,
        grid_size: int,
        magnitude: float,
        num_steps: int,
        num_plots: int,
        num_processes: int,
    ):
        super().__init__(
            "reward_surface_visualization",
            env_factory,
            agent_factory,
            run_dir,
            num_processes,
        )
        self.grid_size = grid_size
        self.magnitude = magnitude
        self.num_steps = num_steps
        self.num_plots = num_plots
        self.num_processes = num_processes
        self.out_dir = run_dir / "analyses" / "reward_surface_visualization"
        self.out_dir.mkdir(exist_ok=True, parents=True)
        self.reward_undiscounted_data_dir = (
            self.out_dir / "reward_surface_undiscounted" / "data"
        )
        self.reward_discounted_data_dir = (
            self.out_dir / "reward_surface_discounted" / "data"
        )
        self.loss_data_dir = self.out_dir / "loss_surface" / "data"
        self.negative_loss_data_dir = self.out_dir / "negative_loss_surface" / "data"

    def _do_analysis(
        self,
        process_pool: torch.multiprocessing.Pool,
        env_step: int,
        overwrite_results: bool,
        show_progress: bool,
    ) -> TensorboardLogs:
        logs = TensorboardLogs()
        for plot_num in range(self.num_plots):
            if (
                not overwrite_results
                and (
                    self.out_dir
                    / "loss"
                    / f"{self._result_filename('loss_surface', env_step, plot_num)}.png"
                ).exists()
            ):
                continue
            if show_progress:
                logger.info(f"Creating plot {plot_num} for step {env_step}.")

            agent = self.agent_factory(self.env_factory())
            direction1 = [
                self.sample_filter_normalized_direction(p.detach())
                for p in agent.policy.parameters()
            ]
            direction2 = [
                self.sample_filter_normalized_direction(p.detach())
                for p in agent.policy.parameters()
            ]

            agent_weights = [p.data.detach() for p in agent.policy.parameters()]
            weights_offsets = [[None] * self.grid_size for _ in range(self.grid_size)]
            coords = np.linspace(-1.0, 1.0, num=self.grid_size)

            for offset1_idx, offset1_scalar in enumerate(coords):
                weights_curr_offset1 = [
                    a_weight + off * offset1_scalar
                    for a_weight, off in zip(agent_weights, direction1)
                ]
                for offset2_idx, offset2_scalar in enumerate(coords):
                    weights_curr_offsets = [
                        a_weight + off * offset2_scalar
                        for a_weight, off in zip(weights_curr_offset1, direction2)
                    ]
                    weights_offsets[offset1_idx][offset2_idx] = weights_curr_offsets

            weights_offsets_flat = [
                item for sublist in weights_offsets for item in sublist
            ]

            analysis_results_iter = process_pool.imap(
                functools.partial(
                    self.analysis_worker,
                    env_factory=self.env_factory,
                    agent_factory=self.agent_factory,
                    num_steps=self.num_steps,
                ),
                weights_offsets_flat,
            )
            analysis_results_flat = [
                r
                for r in tqdm(
                    analysis_results_iter,
                    disable=not show_progress,
                    mininterval=120,
                    total=self.grid_size**2,
                )
            ]

            sampled_projected_gradient_steps = self.sample_projected_gradient_steps(
                direction1, direction2, 10
            )

            plot_info = {
                "env_name": agent.env.envs[0].spec.id,
                "env_step": env_step,
                "plot_num": plot_num,
                "magnitude": self.magnitude,
                "directions": (
                    [d.cpu().numpy() for d in direction1],
                    [d.cpu().numpy() for d in direction2],
                ),
                "sampled_projected_gradient_steps": sampled_projected_gradient_steps,
            }

            rewards_undiscounted = np.array(
                [result.reward_undiscounted for result in analysis_results_flat]
            ).reshape(self.grid_size, self.grid_size)
            self.reward_undiscounted_data_dir.mkdir(parents=True, exist_ok=True)
            rewards_undiscounted_file = (
                self.reward_undiscounted_data_dir
                / f"{self._result_filename('reward_surface_undiscounted', env_step, plot_num)}.pkl"
            )
            with rewards_undiscounted_file.open("wb") as f:
                pickle.dump(plot_info | {"data": rewards_undiscounted}, f)

            rewards_discounted = np.array(
                [result.reward_discounted for result in analysis_results_flat]
            ).reshape(self.grid_size, self.grid_size)
            self.reward_discounted_data_dir.mkdir(parents=True, exist_ok=True)
            rewards_discounted_file = (
                self.reward_discounted_data_dir
                / f"{self._result_filename('rewards_discounted', env_step, plot_num)}.pkl"
            )
            with rewards_discounted_file.open("wb") as f:
                pickle.dump(plot_info | {"data": rewards_discounted}, f)

            loss = np.array(
                [result.ppo_loss for result in analysis_results_flat]
            ).reshape(self.grid_size, self.grid_size)
            self.loss_data_dir.mkdir(parents=True, exist_ok=True)
            loss_file = (
                self.loss_data_dir
                / f"{self._result_filename('loss', env_step, plot_num)}.pkl"
            )
            with loss_file.open("wb") as f:
                pickle.dump(plot_info | {"data": loss}, f)

            self.negative_loss_data_dir.mkdir(parents=True, exist_ok=True)
            negative_loss_file = (
                self.negative_loss_data_dir
                / f"{self._result_filename('negative_loss', env_step, plot_num)}.pkl"
            )
            with negative_loss_file.open("wb") as f:
                pickle.dump(plot_info | {"data": -loss}, f)

            # Plotting needs to happen in a separate process since matplotlib is not thread safe (see
            # https://matplotlib.org/3.1.0/faq/howto_faq.html#working-with-threads)
            logs.update(
                process_pool.apply(
                    functools.partial(
                        plot_all_results,
                        self.out_dir,
                        overwrite=overwrite_results,
                    )
                )
            )
        return logs

    @staticmethod
    def analysis_worker(
        agent_weights: Sequence[torch.Tensor],
        env_factory: Callable[[], gym.Env],
        agent_factory: Callable[
            [Union[gym.Env, stable_baselines3.common.vec_env.VecEnv]],
            stable_baselines3.ppo.PPO,
        ],
        num_steps: int,
    ) -> AnalysisResult:
        env = stable_baselines3.common.vec_env.DummyVecEnv([env_factory])
        agent = agent_factory(env)
        rollout_buffer = stable_baselines3.common.buffers.RolloutBuffer(
            num_steps,
            agent.observation_space,
            agent.action_space,
            "cpu",
            agent.gae_lambda,
            agent.gamma,
        )
        rollout_buffer_no_value_bootstrap = (
            stable_baselines3.common.buffers.RolloutBuffer(
                num_steps,
                agent.observation_space,
                agent.action_space,
                "cpu",
                agent.gae_lambda,
                agent.gamma,
            )
        )
        with torch.no_grad():
            for parameters, weights in zip(agent.policy.parameters(), agent_weights):
                parameters.data[:] = weights
        fill_rollout_buffer(
            agent,
            env,
            rollout_buffer,
            rollout_buffer_no_value_bootstrap,
            show_progress=False,
        )
        episode_rewards_undiscounted = []
        episode_rewards_discounted = []
        curr_reward_undiscounted = None
        curr_reward_discounted = None
        curr_episode_length = 0
        for episode_start, reward in zip(
            rollout_buffer_no_value_bootstrap.episode_starts,
            rollout_buffer_no_value_bootstrap.rewards,
        ):
            if episode_start:
                if curr_episode_length > 0:
                    episode_rewards_undiscounted.append(curr_reward_undiscounted)
                    episode_rewards_discounted.append(curr_reward_discounted)
                curr_episode_length = 0
                curr_reward_discounted = 0.0
                curr_reward_undiscounted = 0.0
            curr_episode_length += 1
            curr_reward_undiscounted += reward
            curr_reward_discounted += agent.gamma ** (curr_episode_length - 1) * reward
        # Since there is no next transition in the buffer, we cannot know if the last episode is complete, so only add
        # the episode if it has the maximum episode length.
        if curr_episode_length == get_episode_length(env.envs[0]):
            episode_rewards_undiscounted.append(curr_reward_undiscounted)
            episode_rewards_discounted.append(curr_reward_discounted)
        mean_episode_reward_undiscounted = np.mean(episode_rewards_undiscounted)
        mean_episode_reward_discounted = np.mean(episode_rewards_discounted)
        loss = ppo_loss(agent, next(rollout_buffer.get()))
        return AnalysisResult(
            mean_episode_reward_undiscounted,
            mean_episode_reward_discounted,
            loss.item(),
        )

    @staticmethod
    def sample_filter_normalized_direction(param: torch.Tensor) -> torch.Tensor:
        ndims = len(param.shape)
        if ndims == 1 or ndims == 0:
            # don't do any random direction for scalars
            return torch.zeros_like(param)
        elif ndims == 2:
            direction = torch.normal(0.0, 1.0, size=param.shape, device=param.device)
            direction /= torch.sqrt(
                torch.sum(torch.square(direction), dim=0, keepdim=True)
            )
            direction *= torch.sqrt(torch.sum(torch.square(param), dim=0, keepdim=True))
            return direction
        elif ndims == 4:
            direction = torch.normal(0.0, 1.0, size=param.shape, device=param.device)
            direction /= torch.sqrt(
                torch.sum(torch.square(direction), dim=(0, 1, 2), keepdim=True)
            )
            direction *= torch.sqrt(
                torch.sum(torch.square(param), dim=(0, 1, 2), keepdim=True)
            )
            return direction
        else:
            raise ValueError(
                f"Only 1, 2, 4 dimensional filters allowed, got {param.shape}."
            )

    def sample_projected_gradient_steps(
        self,
        direction1: Sequence[torch.Tensor],
        direction2: Sequence[torch.Tensor],
        num_samples: int,
    ) -> np.ndarray:
        projected_parameters = []
        for _ in range(num_samples):
            agent = self.agent_factory(self.env_factory())
            rollout_buffer_gradient_step = (
                stable_baselines3.common.buffers.RolloutBuffer(
                    agent.n_steps,
                    agent.observation_space,
                    agent.action_space,
                    agent.device,
                    agent.gae_lambda,
                    agent.gamma,
                )
            )
            fill_rollout_buffer(agent, agent.env, rollout_buffer_gradient_step)
            loss = ppo_loss(
                agent, next(rollout_buffer_gradient_step.get(agent.batch_size))
            )
            agent.policy.zero_grad()
            loss.backward()
            agent.policy.optimizer.step()
            direction1_vec = torch.cat([d.flatten() for d in direction1])
            direction2_vec = torch.cat([d.flatten() for d in direction2])
            directions = torch.stack((direction1_vec, direction2_vec), dim=1)
            new_parameters = torch.cat([p.flatten() for p in agent.policy.parameters()])
            # Projection matrix: (directions^T @ directions)^(-1) @ directions^T
            curr_projected_parameters = torch.linalg.solve(
                directions.T @ directions, directions.T @ new_parameters
            )
            projected_parameters.append(
                curr_projected_parameters.detach().cpu().numpy()
            )
        return np.stack(projected_parameters)

    @staticmethod
    def _result_filename(plot_name: str, env_step: int, plot_idx: int) -> str:
        return f"{plot_name}_{env_step:07d}_{plot_idx:02d}"
