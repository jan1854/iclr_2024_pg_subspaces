import functools
import logging
import pickle
from collections import namedtuple
from pathlib import Path
from typing import Callable, Sequence, Union, Tuple

import gym
import numpy as np
import stable_baselines3.common.base_class
import stable_baselines3.common.buffers
import stable_baselines3.common.vec_env
import torch
from tqdm import tqdm

from action_space_toolbox.analysis.analysis import Analysis
from action_space_toolbox.analysis.reward_surface_visualization.plotting import (
    plot_results,
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

RewardSurfaceAnalysisResult = namedtuple(
    "RewardSurfaceAnalysisResult", "reward_undiscounted reward_discounted"
)
LossSurfaceAnalysisResult = namedtuple(
    "LossSurfaceAnalysisResult", "losses policy_ratios"
)


class RewardSurfaceVisualization(Analysis):
    def __init__(
        self,
        analysis_run_id: str,
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
            analysis_run_id,
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
        self.out_dir = (
            run_dir / "analyses" / "reward_surface_visualization" / analysis_run_id
        )
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
            coords = np.linspace(-self.magnitude, self.magnitude, num=self.grid_size)

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

            reward_surface_results_iter = process_pool.imap(
                functools.partial(
                    self.reward_surface_analysis_worker,
                    env_factory=self.env_factory,
                    agent_factory=self.agent_factory,
                    num_steps=self.num_steps,
                ),
                weights_offsets_flat,
            )

            loss_surface_results = process_pool.apply_async(
                self.loss_surface_analysis_worker,
                args=(
                    weights_offsets,
                    self.env_factory,
                    self.agent_factory,
                    self.num_steps,
                ),
            )

            reward_surface_results_flat = [
                r
                for r in tqdm(
                    reward_surface_results_iter,
                    disable=not show_progress,
                    total=self.grid_size**2,
                )
            ]

            (
                projected_optimizer_steps,
                projected_sgd_steps,
            ) = self.sample_projected_gradient_steps(direction1, direction2, 10)

            loss_surface_results = loss_surface_results.get()

            plot_info = {
                "env_name": agent.env.get_attr("spec")[0].id,
                "env_step": env_step,
                "plot_num": plot_num,
                "magnitude": self.magnitude,
                "directions": (
                    [d.cpu().numpy() for d in direction1],
                    [d.cpu().numpy() for d in direction2],
                ),
                "sampled_projected_optimizer_steps": projected_optimizer_steps,
                "sampled_projected_sgd_steps": projected_sgd_steps,
                "policy_ratio": loss_surface_results.policy_ratios,
            }

            rewards_undiscounted = np.array(
                [result.reward_undiscounted for result in reward_surface_results_flat]
            ).reshape(self.grid_size, self.grid_size)
            self.reward_undiscounted_data_dir.mkdir(parents=True, exist_ok=True)
            rewards_undiscounted_file = (
                self.reward_undiscounted_data_dir
                / f"{self._result_filename('reward_surface_undiscounted', env_step, plot_num)}.pkl"
            )
            with rewards_undiscounted_file.open("wb") as f:
                pickle.dump(plot_info | {"data": rewards_undiscounted}, f)

            rewards_discounted = np.array(
                [result.reward_discounted for result in reward_surface_results_flat]
            ).reshape(self.grid_size, self.grid_size)
            self.reward_discounted_data_dir.mkdir(parents=True, exist_ok=True)
            rewards_discounted_file = (
                self.reward_discounted_data_dir
                / f"{self._result_filename('rewards_discounted', env_step, plot_num)}.pkl"
            )
            with rewards_discounted_file.open("wb") as f:
                pickle.dump(plot_info | {"data": rewards_discounted}, f)

            self.loss_data_dir.mkdir(parents=True, exist_ok=True)
            loss_file = (
                self.loss_data_dir
                / f"{self._result_filename('loss', env_step, plot_num)}.pkl"
            )
            with loss_file.open("wb") as f:
                pickle.dump(plot_info | {"data": loss_surface_results.losses}, f)

            self.negative_loss_data_dir.mkdir(parents=True, exist_ok=True)
            negative_loss_file = (
                self.negative_loss_data_dir
                / f"{self._result_filename('negative_loss', env_step, plot_num)}.pkl"
            )
            with negative_loss_file.open("wb") as f:
                pickle.dump(plot_info | {"data": -loss_surface_results.losses}, f)

            # Plotting needs to happen in a separate process since matplotlib is not thread safe (see
            # https://matplotlib.org/3.1.0/faq/howto_faq.html#working-with-threads)
            logs.update(
                process_pool.apply(
                    functools.partial(
                        plot_results,
                        self.out_dir,
                        step=env_step,
                        plot_num=plot_num,
                        overwrite=overwrite_results,
                    )
                )
            )
        return logs

    @staticmethod
    def reward_surface_analysis_worker(
        agent_weights: Sequence[torch.Tensor],
        env_factory: Callable[[], gym.Env],
        agent_factory: Callable[
            [Union[gym.Env, stable_baselines3.common.vec_env.VecEnv]],
            stable_baselines3.ppo.PPO,
        ],
        num_steps: int,
    ) -> RewardSurfaceAnalysisResult:
        env = stable_baselines3.common.vec_env.DummyVecEnv([env_factory])
        agent = agent_factory(env)
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
            None,
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
        if curr_episode_length == get_episode_length(env):
            episode_rewards_undiscounted.append(curr_reward_undiscounted)
            episode_rewards_discounted.append(curr_reward_discounted)
        mean_episode_reward_undiscounted = np.mean(episode_rewards_undiscounted)
        mean_episode_reward_discounted = np.mean(episode_rewards_discounted)
        return RewardSurfaceAnalysisResult(
            mean_episode_reward_undiscounted, mean_episode_reward_discounted
        )

    @staticmethod
    def loss_surface_analysis_worker(
        all_agent_weights: Sequence[Sequence[Sequence[torch.Tensor]]],
        env_factory: Callable[[], gym.Env],
        agent_factory: Callable[
            [Union[gym.Env, stable_baselines3.common.vec_env.VecEnv]],
            stable_baselines3.ppo.PPO,
        ],
        num_steps: int,
    ) -> LossSurfaceAnalysisResult:
        grid_size = len(all_agent_weights)
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
        fill_rollout_buffer(
            agent,
            env,
            rollout_buffer,
            None,
            show_progress=False,
        )
        losses = []
        policy_ratios = []
        all_agent_weights_flat = [w for ws in all_agent_weights for w in ws]
        for agent_weights in all_agent_weights_flat:
            with torch.no_grad():
                for parameters, weights in zip(
                    agent.policy.parameters(), agent_weights
                ):
                    parameters.data[:] = weights
            loss, policy_ratio = ppo_loss(agent, next(rollout_buffer.get()))
            losses.append(loss.item())
            policy_ratios.append(policy_ratio.item())
        return LossSurfaceAnalysisResult(
            np.array(losses).reshape(grid_size, grid_size),
            np.array(policy_ratios).reshape(grid_size, grid_size),
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
    ) -> Tuple[np.ndarray, np.ndarray]:
        projected_optimizer_parameters = []
        projected_sgd_parameters = []

        direction1_vec = torch.cat([d.flatten() for d in direction1])
        direction2_vec = torch.cat([d.flatten() for d in direction2])
        directions = torch.stack((direction1_vec, direction2_vec), dim=1).double()

        for _ in range(num_samples):
            agent = self.agent_factory(self.env_factory())
            old_parameters = torch.cat(
                [p.flatten().clone() for p in agent.policy.parameters()]
            )
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
            loss, _ = ppo_loss(
                agent, next(rollout_buffer_gradient_step.get(agent.batch_size))
            )
            agent.policy.zero_grad()
            loss.backward()
            gradient = torch.cat([p.grad.flatten() for p in agent.policy.parameters()])
            new_sgd_parameters = (
                old_parameters - agent.policy.optimizer.param_groups[0]["lr"] * gradient
            )
            agent.policy.optimizer.step()
            new_optimizer_parameters = torch.cat(
                [p.flatten() for p in agent.policy.parameters()]
            )
            # Projection matrix: (directions^T @ directions)^(-1) @ directions^T
            curr_projected_optimizer_parameters = torch.linalg.solve(
                directions.T @ directions,
                directions.T @ (new_optimizer_parameters - old_parameters).double(),
            )
            curr_projected_sgd_parameters = torch.linalg.solve(
                directions.T @ directions,
                directions.T @ (new_sgd_parameters - old_parameters).double(),
            )
            projected_optimizer_parameters.append(
                curr_projected_optimizer_parameters.detach().cpu().numpy()
            )
            projected_sgd_parameters.append(
                curr_projected_sgd_parameters.detach().cpu().numpy()
            )
        return np.stack(projected_optimizer_parameters), np.stack(
            projected_sgd_parameters
        )

    @staticmethod
    def _result_filename(plot_name: str, env_step: int, plot_idx: int) -> str:
        return f"{plot_name}_{env_step:07d}_{plot_idx:02d}"
