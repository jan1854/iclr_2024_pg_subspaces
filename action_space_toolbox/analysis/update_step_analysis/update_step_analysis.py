import functools
from pathlib import Path
from typing import Callable, Sequence, Tuple, Union

import gym
import numpy as np
import stable_baselines3.common.buffers
import stable_baselines3.common.vec_env
import torch.multiprocessing

from action_space_toolbox.analysis.analysis import Analysis
from action_space_toolbox.analysis.util import (
    evaluate_agent_losses,
    evaluate_agent_returns,
    LossEvaluationResult,
    ReturnEvaluationResult,
    flatten_parameters,
    evaluate_returns_rollout_buffer,
)
from action_space_toolbox.util.agent_spec import AgentSpec
from action_space_toolbox.util.get_episode_length import get_episode_length
from action_space_toolbox.util.metrics import mean_relative_difference
from action_space_toolbox.util.sb3_training import (
    fill_rollout_buffer,
    sample_update_trajectory,
)
from action_space_toolbox.util.tensorboard_logs import TensorboardLogs


class UpdateStepAnalysis(Analysis):
    def __init__(
        self,
        analysis_run_id: str,
        env_factory: Callable[[], gym.Env],
        agent_spec: AgentSpec,
        run_dir: Path,
        num_steps_true_loss: int,
        num_updates_evaluation: int,
        num_samples_agent_updates: int,
    ):
        super().__init__(
            "update_step_analysis",
            analysis_run_id,
            env_factory,
            agent_spec,
            run_dir,
        )
        self.num_steps_true_loss = num_steps_true_loss
        self.num_steps_evaluation = num_updates_evaluation
        self.num_updates_evaluation = num_updates_evaluation
        self.num_samples_agent_updates = num_samples_agent_updates

    def _do_analysis(
        self,
        env_step: int,
        logs: TensorboardLogs,
        overwrite_results: bool,
        verbose: bool,
    ) -> TensorboardLogs:
        agent = self.agent_spec.create_agent(self.env_factory())
        rollout_buffer_true_loss = stable_baselines3.common.buffers.RolloutBuffer(
            self.num_steps_true_loss,
            agent.observation_space,
            agent.action_space,
            agent.device,
            agent.gae_lambda,
            agent.gamma,
            n_envs=1,
        )
        rollout_buffer_curr_policy_eval = (
            stable_baselines3.common.buffers.RolloutBuffer(
                self.num_steps_true_loss,
                agent.observation_space,
                agent.action_space,
                agent.device,
                agent.gae_lambda,
                agent.gamma,
                n_envs=1,
            )
        )
        fill_rollout_buffer(
            self.env_factory,
            self.agent_spec,
            rollout_buffer_true_loss,
            rollout_buffer_curr_policy_eval,
        )

        len_update_trajectory = agent.n_steps * agent.n_envs // agent.batch_size

        rollout_buffer_agent = stable_baselines3.common.buffers.RolloutBuffer(
            self.num_samples_agent_updates,
            agent.observation_space,
            agent.action_space,
            agent.device,
            agent.gae_lambda,
            agent.gamma,
            n_envs=1,
        )
        last_episode_done = fill_rollout_buffer(
            self.env_factory,
            self.agent_spec,
            rollout_buffer_agent,
            None,
        )

        return_curr_policy = evaluate_returns_rollout_buffer(
            rollout_buffer_curr_policy_eval,
            agent.gamma,
            get_episode_length(agent.env),
            last_episode_done,
        )
        loss_curr_policy = evaluate_agent_losses(agent, rollout_buffer_true_loss)

        sample_update_trajectory_agent = functools.partial(
            sample_update_trajectory,
            self.agent_spec,
            rollout_buffer_agent,
            agent.batch_size,
            len_update_trajectory,
        )
        avg_update_step_length = self.get_average_update_step_length(
            sample_update_trajectory_agent
        )
        sample_update_trajectory_random = functools.partial(
            self._sample_random_update_trajectory,
            [p.detach().clone() for p in agent.policy.parameters()],
            avg_update_step_length,
            len_update_trajectory,
        )
        sample_update_trajectory_true_loss = functools.partial(
            sample_update_trajectory,
            self.agent_spec,
            rollout_buffer_true_loss,
            None,
            len_update_trajectory,
            repeat_data=True,
        )

        (
            loss_single_step_agent,
            returns_single_step_agent,
            loss_trajectory_agent,
            returns_trajectory_agent,
        ) = self.update_step_analysis_worker(
            self.num_updates_evaluation,
            1,
            self.agent_spec,
            sample_update_trajectory_agent,
            rollout_buffer_true_loss,
            self.env_factory,
        )
        (
            loss_single_step_random,
            returns_single_step_random,
            loss_trajectory_random,
            returns_trajectory_random,
        ) = self.update_step_analysis_worker(
            self.num_updates_evaluation,
            1,
            self.agent_spec,
            sample_update_trajectory_random,
            rollout_buffer_true_loss,
            self.env_factory,
        )
        # For the other agent and random configurations, we evaluate self.num_updates_evaluation updates for one
        # episode. Since we only have one "true gradient" update, we evaluate this update for
        # self.num_updates_evaluation episodes (to get the same total number of evaluation episodes).
        (
            loss_single_step_true_loss,
            returns_single_step_true_loss,
            loss_trajectory_true_loss,
            returns_trajectory_true_loss,
        ) = self.update_step_analysis_worker(
            1,
            self.num_updates_evaluation,
            self.agent_spec,
            sample_update_trajectory_true_loss,
            rollout_buffer_true_loss,
            self.env_factory,
        )

        self.log_results(
            "single_update_step/random",
            loss_single_step_random,
            returns_single_step_random,
            loss_curr_policy,
            return_curr_policy,
            env_step,
            logs,
        )
        self.log_results(
            "update_trajectory/random",
            loss_trajectory_random,
            returns_trajectory_random,
            loss_curr_policy,
            return_curr_policy,
            env_step,
            logs,
        )
        self.log_results(
            "single_update_step/true_gradient",
            loss_single_step_true_loss,
            returns_single_step_true_loss,
            loss_curr_policy,
            return_curr_policy,
            env_step,
            logs,
        )
        self.log_results(
            "update_trajectory/true_gradient",
            loss_trajectory_true_loss,
            returns_trajectory_true_loss,
            loss_curr_policy,
            return_curr_policy,
            env_step,
            logs,
        )
        self.log_results(
            "single_update_step/agent",
            loss_single_step_agent,
            returns_single_step_agent,
            loss_curr_policy,
            return_curr_policy,
            env_step,
            logs,
        )
        self.log_results(
            "update_trajectory/agent",
            loss_trajectory_agent,
            returns_trajectory_agent,
            loss_curr_policy,
            return_curr_policy,
            env_step,
            logs,
        )

        plot_names = [
            "combined_loss",
            "policy_loss",
            "value_function_loss",
            "reward_undiscounted",
            "reward_discounted",
        ]
        values_curr_policy = [
            loss_curr_policy.combined_losses.item(),
            loss_curr_policy.policy_losses.item(),
            loss_curr_policy.value_function_losses.item(),
            return_curr_policy.rewards_undiscounted.item(),
            return_curr_policy.rewards_discounted.item(),
        ]
        for plot_name, value in zip(plot_names, values_curr_policy):
            logs.add_scalar(f"zz_raw/original_agent/{plot_name}", value, env_step)

        for plot_name in plot_names:
            logs.add_multiline_scalar(
                f"single_update_step/{plot_name}",
                [
                    f"single_update_step/{update_type}/{plot_name}"
                    for update_type in ["random", "agent", "true_gradient"]
                ],
            )
        for plot_name in plot_names:
            logs.add_multiline_scalar(
                f"update_trajectory/{plot_name}",
                [
                    f"update_trajectory/{update_type}/{plot_name}"
                    for update_type in ["random", "agent", "true_gradient"]
                ],
            )
        return logs

    @classmethod
    def update_step_analysis_worker(
        cls,
        num_update_trajectories: int,
        num_evaluation_episodes: int,
        agent_spec: AgentSpec,
        update_trajectory_sampler: Callable[[], Sequence[torch.Tensor]],
        rollout_buffer_true_loss: stable_baselines3.common.buffers.RolloutBuffer,
        env_factory: Callable[
            [], Union[gym.Env, stable_baselines3.common.vec_env.VecEnv]
        ],
    ) -> Tuple[
        LossEvaluationResult,
        ReturnEvaluationResult,
        LossEvaluationResult,
        ReturnEvaluationResult,
    ]:
        results_loss_single_step = []
        results_loss_trajectory = []
        results_return_single_step = []
        results_return_trajectory = []
        env = env_factory()
        for _ in range(num_update_trajectories):
            update_trajectory = update_trajectory_sampler()
            agent_spec_single_step = agent_spec.copy_with_new_parameters(
                update_trajectory[0]
            )
            agent_single_step = agent_spec_single_step.create_agent()
            agent_spec_trajectory = agent_spec.copy_with_new_parameters(
                update_trajectory[-1]
            )
            agent_trajectory = agent_spec_trajectory.create_agent()
            results_loss_single_step.append(
                evaluate_agent_losses(
                    [agent_single_step],
                    rollout_buffer_true_loss,
                )
            )
            results_loss_trajectory.append(
                evaluate_agent_losses([agent_trajectory], rollout_buffer_true_loss)
            )
            results_return_single_step.append(
                evaluate_agent_returns(
                    [agent_single_step],
                    env,
                    num_episodes=num_evaluation_episodes,
                )
            )
            results_return_trajectory.append(
                evaluate_agent_returns(
                    [agent_trajectory],
                    env,
                    num_episodes=num_evaluation_episodes,
                )
            )
        results_loss_single_step = LossEvaluationResult.concatenate(
            results_loss_single_step
        )
        results_loss_trajectory = LossEvaluationResult.concatenate(
            results_loss_trajectory
        )
        results_return_single_step = ReturnEvaluationResult.concatenate(
            results_return_single_step
        )
        results_return_trajectory = ReturnEvaluationResult.concatenate(
            results_return_trajectory
        )
        return (
            results_loss_single_step,
            results_return_single_step,
            results_loss_trajectory,
            results_return_trajectory,
        )

    @classmethod
    def _sample_random_update_trajectory(
        cls,
        initial_weights: Sequence[torch.Tensor],
        update_step_length: float,
        num_update_steps: int,
    ) -> Sequence[Sequence[torch.Tensor]]:
        """
        Samples an update trajectory of random steps, starting with the given initial parameters. Each step has given
        length.
        :param initial_weights:             The initial weights (the weights of agent before the update)
        :param update_step_length:          The length of each update step
        :param num_update_steps:            The length of the trajectory in update steps
        :return:                            Ab update trajectory that take random steps of the given length
        """
        device = initial_weights[0].device
        random_update_trajectory = []
        new_weights = initial_weights
        for _ in range(num_update_steps):
            random_step_unnormalized = [
                torch.distributions.uniform.Uniform(-1, 1)
                .sample(layer_step.shape)
                .to(device)
                for layer_step in initial_weights
            ]
            curr_step_size = torch.norm(flatten_parameters(random_step_unnormalized))
            random_step = [
                layer_step * update_step_length / curr_step_size
                for layer_step in random_step_unnormalized
            ]
            new_weights = [w + s for w, s in zip(new_weights, random_step)]
            random_update_trajectory.append(new_weights)
        return random_update_trajectory

    @classmethod
    def log_results(
        cls,
        name: str,
        losses: LossEvaluationResult,
        returns: ReturnEvaluationResult,
        loss_curr_policy: LossEvaluationResult,
        return_curr_policy: ReturnEvaluationResult,
        env_step: int,
        logs: TensorboardLogs,
    ) -> None:
        plot_names = [
            "combined_loss",
            "policy_loss",
            "value_function_loss",
            "reward_undiscounted",
            "reward_discounted",
        ]

        values_original = [
            loss_curr_policy.combined_losses.item(),
            loss_curr_policy.policy_losses.item(),
            loss_curr_policy.value_function_losses.item(),
            return_curr_policy.rewards_undiscounted.item(),
            return_curr_policy.rewards_discounted.item(),
        ]
        values_after_update = [
            losses.combined_losses,
            losses.policy_losses,
            losses.value_function_losses,
            returns.rewards_undiscounted,
            returns.rewards_discounted,
        ]

        for plot_name, value_after_update, value_original in zip(
            plot_names, values_after_update, values_original
        ):
            logs.add_scalar(
                f"absolute_difference/{name}/{plot_name}",
                np.mean(value_after_update - value_original),  # type: ignore
                env_step,
            )
            logs.add_scalar(
                f"relative_difference/{name}/{plot_name}",
                mean_relative_difference(value_original, value_after_update),
                env_step,
            )
            logs.add_scalar(
                f"zz_raw/{name}/{plot_name}",
                np.mean(value_after_update),  # type: ignore
                env_step,
            )

    @classmethod
    def get_average_update_step_length(
        cls,
        update_trajectory_sampler: Callable[[], Sequence[torch.Tensor]],
        num_trajectories: int = 20,
    ) -> float:
        update_trajectories = [
            update_trajectory_sampler() for _ in range(num_trajectories)
        ]
        update_trajectories = torch.stack(
            [
                torch.stack([flatten_parameters(step) for step in update_traj])
                for update_traj in update_trajectories
            ]
        )

        update_trajectories_difference = (
            update_trajectories[:, 1:] - update_trajectories[:, :-1]
        )
        avg_step_size_agent = torch.mean(
            torch.norm(update_trajectories_difference, dim=-1)
        )
        return avg_step_size_agent.item()
