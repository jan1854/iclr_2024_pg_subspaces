import functools
import itertools
from pathlib import Path
from typing import Callable, Sequence

import gym
import numpy as np
import stable_baselines3.common.buffers
import torch.multiprocessing

from action_space_toolbox.analysis.analysis import Analysis
from action_space_toolbox.util.agent_spec import AgentSpec
from action_space_toolbox.util.sb3_training import (
    fill_rollout_buffer,
    sample_update_trajectory,
    evaluate_agent_returns,
    flatten_parameters,
    evaluate_agent_losses,
    AgentLosses,
    AgentReturns,
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
        num_steps_evaluation: int,
        num_update_trajectories: int,
    ):
        super().__init__(
            "update_step_analysis",
            analysis_run_id,
            env_factory,
            agent_spec,
            run_dir,
            num_processes=1,
        )
        self.num_steps_true_loss = num_steps_true_loss
        self.num_steps_evaluation = num_steps_evaluation
        self.num_update_trajectories = num_update_trajectories

    def _do_analysis(
        self,
        process_pool: torch.multiprocessing.Pool,
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
        fill_rollout_buffer(
            self.env_factory,
            self.agent_spec,
            rollout_buffer_true_loss,
            None,
            self.num_processes,
        )

        len_update_trajectory = agent.n_steps * agent.n_envs // agent.batch_size
        update_trajectory_true_loss = sample_update_trajectory(
            rollout_buffer_true_loss,
            self.agent_spec.create_agent(self.env_factory()),
            agent.policy.optimizer,
            None,
            len_update_trajectory,
            repeat_data=True,
        )

        rollout_buffer_agent = stable_baselines3.common.buffers.RolloutBuffer(
            agent.n_steps * agent.n_envs,
            agent.observation_space,
            agent.action_space,
            agent.device,
            agent.gae_lambda,
            agent.gamma,
            n_envs=1,
        )
        update_trajectories_agent = []
        for _ in range(self.num_update_trajectories):
            rollout_buffer_agent.reset()
            fill_rollout_buffer(
                self.env_factory,
                self.agent_spec,
                rollout_buffer_agent,
                None,
                self.num_processes,
            )
            update_trajectories_agent.append(
                sample_update_trajectory(
                    rollout_buffer_agent,
                    self.agent_spec.create_agent(self.env_factory()),
                    agent.policy.optimizer,
                    agent.batch_size,
                    len_update_trajectory,
                )
            )

        update_trajectories_random = self._sample_random_update_trajectories(
            [p.detach() for p in agent.policy.parameters()], update_trajectories_agent
        )

        agent_spec_single_step_true_loss = self.agent_spec.copy_with_new_weights(
            update_trajectory_true_loss[0]
        )
        agent_specs_single_step_agent = [
            self.agent_spec.copy_with_new_weights(traj[0])
            for traj in update_trajectories_agent
        ]
        agent_specs_single_step_random = [
            self.agent_spec.copy_with_new_weights(traj[0])
            for traj in update_trajectories_random
        ]
        agent_spec_trajectory_true_loss = self.agent_spec.copy_with_new_weights(
            update_trajectory_true_loss[-1]
        )
        agent_specs_trajectory_agent = [
            self.agent_spec.copy_with_new_weights(traj[-1])
            for traj in update_trajectories_agent
        ]
        agent_specs_trajectory_random = [
            self.agent_spec.copy_with_new_weights(traj[-1])
            for traj in update_trajectories_random
        ]

        evaluate_agent_common_parameters = functools.partial(
            evaluate_agent_returns,
            env_factory=self.env_factory,
            num_steps=self.num_steps_evaluation,
        )
        returns_single_step_true_loss = process_pool.apply_async(
            evaluate_agent_common_parameters, ([agent_spec_single_step_true_loss],)
        )
        returns_single_step_agent = process_pool.map_async(
            evaluate_agent_common_parameters, agent_specs_single_step_agent
        )
        returns_single_step_random = process_pool.map_async(
            evaluate_agent_common_parameters, agent_specs_single_step_random
        )
        returns_trajectory_true_loss = process_pool.apply_async(
            evaluate_agent_common_parameters, ([agent_spec_trajectory_true_loss],)
        )
        returns_trajectory_agent = process_pool.map_async(
            evaluate_agent_common_parameters, agent_specs_trajectory_agent
        )
        returns_trajectory_random = process_pool.map_async(
            evaluate_agent_common_parameters, agent_specs_trajectory_random
        )

        loss_single_step_true_loss = evaluate_agent_losses(
            [agent_spec_trajectory_true_loss],
            rollout_buffer_true_loss,
            self.env_factory,
        )
        loss_single_step_agent = evaluate_agent_losses(
            agent_specs_single_step_agent,
            rollout_buffer_true_loss,
            self.env_factory,
        )
        loss_single_step_random = evaluate_agent_losses(
            agent_specs_single_step_random,
            rollout_buffer_true_loss,
            self.env_factory,
        )
        loss_trajectory_true_loss = evaluate_agent_losses(
            [agent_spec_trajectory_true_loss],
            rollout_buffer_true_loss,
            self.env_factory,
        )
        loss_trajectory_agent = evaluate_agent_losses(
            agent_specs_trajectory_agent,
            rollout_buffer_true_loss,
            self.env_factory,
        )
        loss_trajectory_random = evaluate_agent_losses(
            agent_specs_trajectory_random,
            rollout_buffer_true_loss,
            self.env_factory,
        )

        returns_single_step_true_loss = returns_single_step_true_loss.get()
        returns_single_step_agent = AgentReturns.concatenate(
            returns_single_step_agent.get()
        )
        returns_single_step_random = AgentReturns.concatenate(
            returns_single_step_random.get()
        )
        returns_trajectory_true_loss = returns_trajectory_true_loss.get()
        returns_trajectory_agent = AgentReturns.concatenate(
            returns_trajectory_agent.get()
        )
        returns_trajectory_random = AgentReturns.concatenate(
            returns_trajectory_random.get()
        )

        self.log_results(
            "true_gradient_single_update_step",
            loss_single_step_true_loss,
            returns_single_step_true_loss,
            env_step,
            logs,
        )
        self.log_results(
            "random_single_update_step",
            loss_single_step_random,
            returns_single_step_random,
            env_step,
            logs,
        )
        self.log_results(
            "agent_single_update_step",
            loss_single_step_agent,
            returns_single_step_agent,
            env_step,
            logs,
        )
        self.log_results(
            "true_gradient_update_trajectory",
            loss_trajectory_true_loss,
            returns_trajectory_true_loss,
            env_step,
            logs,
        )
        self.log_results(
            "random_update_trajectory",
            loss_trajectory_random,
            returns_trajectory_random,
            env_step,
            logs,
        )
        self.log_results(
            "agent_update_trajectory",
            loss_trajectory_agent,
            returns_trajectory_agent,
            env_step,
            logs,
        )
        return logs

    @classmethod
    def _sample_random_update_trajectories(
        cls,
        initial_weights: Sequence[torch.Tensor],
        update_trajectories_agent: Sequence[Sequence[Sequence[torch.Tensor]]],
    ) -> Sequence[Sequence[Sequence[torch.Tensor]]]:
        """
        Samples update trajectories of random steps, starting with the given initial parameters. Each step has the same
        length as the average step in the given update trajectory. The number of trajectories and the length of each
        trajectory is the same as for the given update trajectories.
        :param initial_weights:             The initial weights (the weights of agent before the update)
        :param update_trajectories_agent:   Update trajectories resulting from the agent optimization
        :return:                            Update trajectories that take random steps with length being equal to the
                                            average length of the agent optimization steps
        """
        device = initial_weights[0].device
        update_trajectories_with_init = torch.stack(
            [
                flatten_parameters(step)
                for update_traj in update_trajectories_agent
                for step in itertools.chain([initial_weights], update_traj)
            ]
        )
        update_trajectories_difference = (
            update_trajectories_with_init[:, 1:] - update_trajectories_with_init[:, :-1]
        )
        avg_step_size_agent = torch.mean(
            torch.norm(update_trajectories_difference, dim=-1)
        )
        random_update_trajectories = []
        num_trajectories = len(update_trajectories_agent)
        len_trajectories = len(update_trajectories_agent[0])
        for _ in range(num_trajectories):
            random_update_trajectories.append([])
            new_weights = initial_weights
            for _ in range(len_trajectories):
                random_step_unnormalized = [
                    torch.distributions.uniform.Uniform(-1, 1)
                    .sample(layer_step.shape)
                    .to(device)
                    for layer_step in update_trajectories_agent[0][0]
                ]
                curr_step_size = torch.norm(
                    flatten_parameters(random_step_unnormalized)
                )
                random_step = [
                    layer_step * avg_step_size_agent / curr_step_size
                    for layer_step in random_step_unnormalized
                ]
                new_weights = [w + s for w, s in zip(new_weights, random_step)]
                random_update_trajectories[-1].append(new_weights)
        return random_update_trajectories

    @classmethod
    def log_results(
        cls,
        name: str,
        losses: AgentLosses,
        returns: AgentReturns,
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
        values = [
            np.mean(losses.combined_losses),
            np.mean(losses.policy_losses),
            np.mean(losses.value_function_losses),
            np.mean(returns.rewards_undiscounted),
            np.mean(returns.rewards_discounted),
        ]
        for plot_name, value in zip(plot_names, values):
            logs.add_scalar(f"{name}/{plot_name}", value, env_step)  # type: ignore
