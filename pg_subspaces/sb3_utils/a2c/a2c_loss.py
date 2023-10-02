from typing import Tuple

import gym
import stable_baselines3
import stable_baselines3.common.buffers
import torch


def a2c_loss(
    agent: stable_baselines3.ppo.PPO,
    rollout_data: stable_baselines3.common.buffers.RolloutBufferSamples,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    actions = rollout_data.actions
    if isinstance(agent.action_space, gym.spaces.Discrete):
        # Convert discrete action from float to long
        actions = rollout_data.actions.long().flatten()
    values, log_prob, entropy = agent.policy.evaluate_actions(
        rollout_data.observations, actions
    )
    values = values.flatten()

    # Normalize advantage (not present in the original implementation)
    advantages = rollout_data.advantages
    if agent.normalize_advantage:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # Policy gradient loss
    policy_loss = -(advantages * log_prob).mean()

    # Value loss using the TD(gae_lambda) target
    value_loss = torch.nn.functional.mse_loss(rollout_data.returns, values)

    # Entropy loss favor exploration
    if entropy is None:
        # Approximate entropy when no analytical form
        entropy_loss = -torch.mean(-log_prob)
    else:
        entropy_loss = -torch.mean(entropy)

    loss = policy_loss + agent.ent_coef * entropy_loss + agent.vf_coef * value_loss
    return (
        loss,
        policy_loss + agent.ent_coef * entropy_loss,
        agent.vf_coef * value_loss,
    )
