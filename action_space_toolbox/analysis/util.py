import dataclasses
from typing import Callable, Iterable, Optional, Sequence, Union

import gym
import numpy as np
import stable_baselines3.common.buffers
import torch

from action_space_toolbox.util.agent_spec import AgentSpec
from action_space_toolbox.util.get_episode_length import get_episode_length
from action_space_toolbox.util.sb3_training import (
    ppo_loss,
    fill_rollout_buffer,
    maybe_create_agent,
)


@dataclasses.dataclass
class ReturnEvaluationResult:
    rewards_undiscounted: np.ndarray
    rewards_discounted: np.ndarray

    def reshape(self, *dims) -> "ReturnEvaluationResult":
        return ReturnEvaluationResult(
            self.rewards_undiscounted.reshape(*dims),
            self.rewards_discounted.reshape(*dims),
        )

    @classmethod
    def concatenate(
        cls, seq: Sequence["ReturnEvaluationResult"]
    ) -> "ReturnEvaluationResult":
        return ReturnEvaluationResult(
            np.concatenate([ar.rewards_undiscounted for ar in seq]),
            np.concatenate([ar.rewards_discounted for ar in seq]),
        )


@dataclasses.dataclass
class LossEvaluationResult:
    policy_losses: np.ndarray
    value_function_losses: np.ndarray
    combined_losses: np.ndarray
    policy_ratios: np.ndarray

    def reshape(self, *dims) -> "LossEvaluationResult":
        return LossEvaluationResult(
            self.policy_losses.reshape(*dims),
            self.value_function_losses.reshape(*dims),
            self.combined_losses.reshape(*dims),
            self.policy_ratios.reshape(*dims),
        )

    @classmethod
    def concatenate(
        cls, seq: Sequence["LossEvaluationResult"]
    ) -> "LossEvaluationResult":
        return LossEvaluationResult(
            np.concatenate([al.policy_losses for al in seq]),
            np.concatenate([al.value_function_losses for al in seq]),
            np.concatenate([al.combined_losses for al in seq]),
            np.concatenate([al.policy_ratios for al in seq]),
        )


def evaluate_agent_returns(
    agents_or_specs: Union[
        AgentSpec,
        Sequence[AgentSpec],
        stable_baselines3.ppo.PPO,
        Sequence[stable_baselines3.ppo.PPO],
    ],
    env_or_factory: Union[gym.Env, Callable[[], gym.Env]],
    num_steps: Optional[int] = None,
    num_epsiodes: Optional[int] = None,
) -> ReturnEvaluationResult:
    assert (num_steps is not None) ^ (
        num_epsiodes is not None
    ), "Exactly one of num_steps or num_episodes must be specified."
    if isinstance(env_or_factory, Callable):
        env = env_or_factory()
    else:
        env = env_or_factory
    if num_epsiodes is not None:
        num_steps = num_epsiodes * get_episode_length(env)
    results = []
    if not isinstance(agents_or_specs, Iterable):
        agents_or_specs = [agents_or_specs]
    for agent_or_spec in agents_or_specs:
        agent = maybe_create_agent(agent_or_spec)
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

        fill_rollout_buffer(
            env_or_factory,
            agent_or_spec,
            None,
            rollout_buffer_no_value_bootstrap,
        )

        results.append(
            evaluate_returns_rollout_buffer(
                rollout_buffer_no_value_bootstrap, agent.gamma, get_episode_length(env)
            )
        )

    return ReturnEvaluationResult.concatenate(results)


def evaluate_returns_rollout_buffer(
    rollout_buffer_no_value_bootstrap: stable_baselines3.common.buffers.RolloutBuffer,
    discount_factor: float,
    episode_length: Optional[int],
) -> ReturnEvaluationResult:
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
        curr_reward_discounted += discount_factor ** (curr_episode_length - 1) * reward
    # Since there is no next transition in the buffer, we cannot know if the last episode is complete, so only add
    # the episode if it has the maximum episode length.
    if episode_length is not None and curr_episode_length == episode_length:
        episode_rewards_undiscounted.append(curr_reward_undiscounted)
        episode_rewards_discounted.append(curr_reward_discounted)
    return ReturnEvaluationResult(
        np.mean(episode_rewards_undiscounted, keepdims=True),
        np.mean(episode_rewards_discounted, keepdims=True),
    )


def evaluate_agent_losses(
    agents_or_specs: Union[
        AgentSpec,
        Sequence[AgentSpec],
        stable_baselines3.ppo.PPO,
        Sequence[stable_baselines3.ppo.PPO],
    ],
    rollout_buffer: stable_baselines3.common.buffers.RolloutBuffer,
) -> LossEvaluationResult:
    combined_losses = []
    policy_losses = []
    value_function_losses = []
    policy_ratios = []
    if not isinstance(agents_or_specs, Iterable):
        agents_or_specs = [agents_or_specs]
    for agent_or_spec in agents_or_specs:
        agent = maybe_create_agent(agent_or_spec)
        (
            combined_loss,
            policy_loss,
            value_function_loss,
            policy_ratio,
        ) = ppo_loss(agent, next(rollout_buffer.get()))
        combined_losses.append(combined_loss.item())
        policy_losses.append(policy_loss.item())
        value_function_losses.append(value_function_loss.item())
        policy_ratios.append(policy_ratio.item())
    return LossEvaluationResult(
        np.array(policy_losses, dtype=np.float32),
        np.array(value_function_losses, dtype=np.float32),
        np.array(combined_losses, dtype=np.float32),
        np.array(policy_ratios, dtype=np.float32),
    )


def flatten_parameters(seq: Sequence[torch.Tensor]) -> torch.Tensor:
    return torch.cat([s.flatten() for s in seq])
