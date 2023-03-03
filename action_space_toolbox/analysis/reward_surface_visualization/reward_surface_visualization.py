import functools
import logging
import pickle
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence, Tuple

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
from action_space_toolbox.analysis.util import (
    flatten_parameters,
    evaluate_agent_returns,
    evaluate_agent_losses,
)
from action_space_toolbox.util.agent_spec import AgentSpec
from action_space_toolbox.util.sb3_training import (
    fill_rollout_buffer,
    ppo_loss,
    ppo_gradient,
)
from action_space_toolbox.util.tensorboard_logs import TensorboardLogs

logger = logging.getLogger(__name__)


class RewardSurfaceVisualization(Analysis):
    def __init__(
        self,
        analysis_run_id: str,
        env_factory: Callable[[], gym.Env],
        agent_spec: AgentSpec,
        run_dir: Path,
        grid_size: int,
        magnitude: float,
        plot_in_gradient_direction: bool,
        num_samples_true_loss: int,
        num_steps: int,
        num_plots: int,
        num_processes: int,
        plot_sgd_steps: bool,
        plot_true_gradient_steps: bool,
        max_gradient_trajectories: int,
        max_steps_per_gradient_trajectory: Optional[int],
    ):
        super().__init__(
            "reward_surface_visualization",
            analysis_run_id,
            env_factory,
            agent_spec,
            run_dir,
            num_processes,
        )
        self.grid_size = grid_size
        self.magnitude = magnitude
        self.plot_in_gradient_direction = plot_in_gradient_direction
        self.num_samples_true_loss = num_samples_true_loss
        self.num_steps = num_steps
        self.num_plots = num_plots
        self.num_processes = num_processes
        self.plot_sgd_steps = plot_sgd_steps
        self.plot_true_gradient_steps = plot_true_gradient_steps
        self.max_gradient_trajectories = max_gradient_trajectories
        self.max_steps_per_gradient_trajectory = max_steps_per_gradient_trajectory
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
        self.policy_loss_data_dir = self.out_dir / "policy_loss_surface" / "data"
        self.negative_policy_loss_data_dir = (
            self.out_dir / "negative_policy_loss_surface" / "data"
        )
        self.vf_loss_data_dir = self.out_dir / "value_function_loss_surface" / "data"
        self.negative_vf_loss_data_dir = (
            self.out_dir / "negative_value_function_loss_surface" / "data"
        )
        self.loss_data_dir = self.out_dir / "loss_surface" / "data"
        self.negative_loss_data_dir = self.out_dir / "negative_loss_surface" / "data"

    def _do_analysis(
        self,
        process_pool: torch.multiprocessing.Pool,
        env_step: int,
        logs: TensorboardLogs,
        overwrite_results: bool,
        show_progress: bool,
    ) -> TensorboardLogs:
        for plot_num in range(self.num_plots):
            if (
                not overwrite_results
                and (
                    self.out_dir
                    / "loss_surface"
                    / f"{self._result_filename('loss_surface', env_step, plot_num)}.png"
                ).exists()
            ):
                continue
            if show_progress:
                logger.info(f"Creating plot {plot_num} for step {env_step}.")

            agent = self.agent_spec.create_agent(self.env_factory())
            rollout_buffer_true_loss = stable_baselines3.common.buffers.RolloutBuffer(
                self.num_samples_true_loss,
                agent.observation_space,
                agent.action_space,
                agent.device,
                agent.gae_lambda,
                agent.gamma,
            )
            fill_rollout_buffer(
                self.env_factory,
                self.agent_spec,
                rollout_buffer_true_loss,
                num_spawned_processes=self.num_processes,
            )

            direction2 = [
                self.sample_filter_normalized_direction(p.detach())
                for p in agent.policy.parameters()
            ]

            if self.plot_in_gradient_direction:
                # Normalize the gradient to have the same length as the random direction vector (as described in
                # appendix I of (Sullivan, 2022: Cliff Diving: Exploring Reward Surfaces in Reinforcement Learning
                # Environments))
                gradient, _, _ = ppo_gradient(
                    agent, next(rollout_buffer_true_loss.get())
                )
                gradient_norm = torch.linalg.norm(
                    torch.cat([g.flatten() for g in gradient])
                )
                direction2_norm = torch.linalg.norm(
                    torch.cat([g.flatten() for g in direction2])
                )
                direction1 = [g / gradient_norm * direction2_norm for g in gradient]
            else:
                direction1 = [
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

            agent_specs = [
                [self.agent_spec.copy_with_new_weights(weights) for weights in sublist]
                for sublist in weights_offsets
            ]

            agent_specs_flat = [
                agent_spec for sublist in agent_specs for agent_spec in sublist
            ]

            reward_surface_results_iter = process_pool.imap(
                functools.partial(
                    evaluate_agent_returns,
                    env_or_factory=self.env_factory,
                    num_steps=self.num_steps,
                ),
                [[agent_spec] for agent_spec in agent_specs_flat],
            )

            loss_surface_results = evaluate_agent_losses(
                agent_specs_flat, rollout_buffer_true_loss
            ).reshape(self.grid_size, self.grid_size)

            (
                projected_optimizer_steps,
                projected_sgd_steps,
                projected_optimizer_steps_true_grad,
                projected_sgd_steps_true_grad,
            ) = self.sample_projected_update_trajectories(
                direction1,
                direction2,
                max(10, self.max_gradient_trajectories),
                rollout_buffer_true_loss,
            )

            reward_surface_results_flat = [
                r
                for r in tqdm(
                    reward_surface_results_iter,
                    disable=not show_progress,
                    total=self.grid_size**2,
                )
            ]

            plot_info = {
                "env_name": agent.env.get_attr("spec")[0].id,
                "env_step": env_step,
                "plot_num": plot_num,
                "magnitude": self.magnitude,
                "directions": (
                    [d.cpu().numpy() for d in direction1],
                    [d.cpu().numpy() for d in direction2],
                ),
                "gradient_direction": 0 if self.plot_in_gradient_direction else None,
                "num_samples_true_loss": self.num_samples_true_loss,
                "sampled_projected_optimizer_steps": projected_optimizer_steps,
                "sampled_projected_sgd_steps": projected_sgd_steps,
                "sampled_projected_optimizer_steps_true_gradient": projected_optimizer_steps_true_grad,
                "sampled_projected_sgd_steps_true_gradient": projected_sgd_steps_true_grad,
                "policy_ratio": loss_surface_results.policy_ratios,
            }

            rewards_undiscounted = np.array(
                [result.rewards_undiscounted for result in reward_surface_results_flat]
            ).reshape(self.grid_size, self.grid_size)
            self.reward_undiscounted_data_dir.mkdir(parents=True, exist_ok=True)
            rewards_undiscounted_file = (
                self.reward_undiscounted_data_dir
                / f"{self._result_filename('reward_surface_undiscounted', env_step, plot_num)}.pkl"
            )
            with rewards_undiscounted_file.open("wb") as f:
                pickle.dump(plot_info | {"data": rewards_undiscounted}, f)

            rewards_discounted = np.array(
                [result.rewards_discounted for result in reward_surface_results_flat]
            ).reshape(self.grid_size, self.grid_size)
            self.reward_discounted_data_dir.mkdir(parents=True, exist_ok=True)
            rewards_discounted_file = (
                self.reward_discounted_data_dir
                / f"{self._result_filename('rewards_discounted', env_step, plot_num)}.pkl"
            )
            with rewards_discounted_file.open("wb") as f:
                pickle.dump(plot_info | {"data": rewards_discounted}, f)

            self.policy_loss_data_dir.mkdir(parents=True, exist_ok=True)
            policy_loss_file = (
                self.policy_loss_data_dir
                / f"{self._result_filename('policy_loss', env_step, plot_num)}.pkl"
            )
            with policy_loss_file.open("wb") as f:
                pickle.dump(plot_info | {"data": loss_surface_results.policy_losses}, f)

            self.negative_policy_loss_data_dir.mkdir(parents=True, exist_ok=True)
            negative_policy_loss_file = (
                self.negative_policy_loss_data_dir
                / f"{self._result_filename('negative_policy_loss', env_step, plot_num)}.pkl"
            )
            with negative_policy_loss_file.open("wb") as f:
                pickle.dump(
                    plot_info | {"data": -loss_surface_results.policy_losses}, f
                )

            self.vf_loss_data_dir.mkdir(parents=True, exist_ok=True)
            vf_loss_file = (
                self.vf_loss_data_dir
                / f"{self._result_filename('value_function_loss', env_step, plot_num)}.pkl"
            )
            with vf_loss_file.open("wb") as f:
                pickle.dump(
                    plot_info | {"data": loss_surface_results.value_function_losses}, f
                )

            self.negative_vf_loss_data_dir.mkdir(parents=True, exist_ok=True)
            negative_vf_loss_file = (
                self.negative_vf_loss_data_dir
                / f"{self._result_filename('negative_value_function_loss', env_step, plot_num)}.pkl"
            )
            with negative_vf_loss_file.open("wb") as f:
                pickle.dump(
                    plot_info | {"data": -loss_surface_results.value_function_losses}, f
                )

            self.loss_data_dir.mkdir(parents=True, exist_ok=True)
            loss_file = (
                self.loss_data_dir
                / f"{self._result_filename('loss', env_step, plot_num)}.pkl"
            )
            with loss_file.open("wb") as f:
                pickle.dump(
                    plot_info | {"data": loss_surface_results.combined_losses}, f
                )

            self.negative_loss_data_dir.mkdir(parents=True, exist_ok=True)
            negative_loss_file = (
                self.negative_loss_data_dir
                / f"{self._result_filename('negative_loss', env_step, plot_num)}.pkl"
            )
            with negative_loss_file.open("wb") as f:
                pickle.dump(
                    plot_info | {"data": -loss_surface_results.combined_losses}, f
                )

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
                        plot_sgd_steps=self.plot_sgd_steps,
                        plot_true_gradient_steps=self.plot_true_gradient_steps,
                        max_gradient_trajectories=self.max_gradient_trajectories,
                        max_steps_per_gradient_trajectory=self.max_steps_per_gradient_trajectory,
                    )
                )
            )
        return logs

    @classmethod
    def sample_filter_normalized_direction(cls, param: torch.Tensor) -> torch.Tensor:
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

    def sample_projected_update_trajectories(
        self,
        direction1: Sequence[torch.Tensor],
        direction2: Sequence[torch.Tensor],
        num_samples: int,
        rollout_buffer_true_gradient: stable_baselines3.common.buffers.RolloutBuffer,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        projected_optimizer_parameters = []
        projected_sgd_parameters = []

        direction1_vec = torch.cat([d.flatten() for d in direction1])
        direction2_vec = torch.cat([d.flatten() for d in direction2])
        directions = torch.stack((direction1_vec, direction2_vec), dim=1).double()

        for sample in range(num_samples):
            agent = self.agent_spec.create_agent(self.env_factory())
            rollout_buffer_gradient_step = (
                stable_baselines3.common.buffers.RolloutBuffer(
                    agent.n_steps,
                    agent.observation_space,
                    agent.action_space,
                    agent.device,
                    agent.gae_lambda,
                    agent.gamma,
                    agent.n_envs,
                )
            )
            fill_rollout_buffer(
                self.env_factory, self.agent_spec, rollout_buffer_gradient_step
            )

            data = list(rollout_buffer_gradient_step.get(agent.batch_size))
            agent = self.agent_spec.create_agent(self.env_factory())
            curr_proj_optimizer_parameters = self._sample_projected_update_trajectory(
                data,
                agent,
                agent.policy.optimizer,
                directions,
            )
            projected_optimizer_parameters.append(curr_proj_optimizer_parameters)

            agent = self.agent_spec.create_agent(self.env_factory())
            sgd = torch.optim.SGD(
                agent.policy.parameters(), agent.policy.optimizer.param_groups[0]["lr"]
            )
            curr_proj_sgd_parameters = self._sample_projected_update_trajectory(
                data,
                agent,
                sgd,
                directions,
            )
            projected_sgd_parameters.append(curr_proj_sgd_parameters)

        agent = self.agent_spec.create_agent(self.env_factory())
        true_gradient_data = [next(rollout_buffer_true_gradient.get())] * 32
        projected_optimizer_parameters_true_grad = (
            self._sample_projected_update_trajectory(
                true_gradient_data,
                agent,
                agent.policy.optimizer,
                directions,
            )
        )

        agent = self.agent_spec.create_agent(self.env_factory())
        sgd = torch.optim.SGD(
            agent.policy.parameters(), agent.policy.optimizer.param_groups[0]["lr"]
        )
        projected_sgd_parameters_true_grad = self._sample_projected_update_trajectory(
            true_gradient_data,
            agent,
            sgd,
            directions,
        )

        return (
            np.stack(projected_optimizer_parameters),
            np.stack(projected_sgd_parameters),
            projected_optimizer_parameters_true_grad[None],
            projected_sgd_parameters_true_grad[None],
        )

    @classmethod
    def _sample_projected_update_trajectory(
        cls,
        data: Iterable[stable_baselines3.common.buffers.RolloutBufferSamples],
        agent: stable_baselines3.ppo.PPO,
        optimizer: torch.optim.Optimizer,
        directions: torch.Tensor,
    ) -> np.ndarray:
        projected_parameters = []
        old_parameters = flatten_parameters(agent.policy.parameters())
        for batch in data:
            loss, _, _, _ = ppo_loss(agent, batch)
            agent.policy.zero_grad()
            loss.backward()
            optimizer.step()
            new_optimizer_parameters = flatten_parameters(agent.policy.parameters())
            # Projection matrix: (directions^T @ directions)^(-1) @ directions^T
            curr_proj_params = torch.linalg.solve(
                directions.T @ directions,
                directions.T @ (new_optimizer_parameters - old_parameters).double(),
            ).to(torch.float32)
            projected_parameters.append(curr_proj_params.detach().cpu().numpy())
        return np.stack(projected_parameters)

    @classmethod
    def _result_filename(cls, plot_name: str, env_step: int, plot_idx: int) -> str:
        return f"{plot_name}_{env_step:07d}_{plot_idx:02d}"
