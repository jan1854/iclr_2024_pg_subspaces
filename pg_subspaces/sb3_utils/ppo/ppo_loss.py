from typing import Tuple

import gym
import stable_baselines3
import stable_baselines3.common.buffers
import torch


def ppo_loss(
    agent: stable_baselines3.ppo.PPO,
    rollout_data: stable_baselines3.common.buffers.RolloutBufferSamples,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Calculates PPO's policy loss. This code is copied from stables-baselines3.PPO.train(). Needs to be copied since
    in the PPO implementation, the loss is not extracted to a separate method.

    :param rollout_data:
    :return:
    """
    assert isinstance(agent, stable_baselines3.ppo.PPO)

    # Compute current clip range
    clip_range = agent.clip_range(agent._current_progress_remaining)
    # Optional: clip range for the value function
    if agent.clip_range_vf is not None:
        clip_range_vf = agent.clip_range_vf(agent._current_progress_remaining)

    actions = rollout_data.actions
    if isinstance(agent.action_space, gym.spaces.Discrete):
        # Convert discrete action from float to long
        actions = rollout_data.actions.long().flatten()

    # Re-sample the noise matrix because the log_std has changed
    if agent.use_sde:
        agent.policy.reset_noise(agent.batch_size)

    values, log_prob, entropy = agent.policy.evaluate_actions(
        rollout_data.observations, actions
    )
    values = values.flatten()
    # Normalize advantage
    advantages = rollout_data.advantages
    if agent.normalize_advantage:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # ratio between old and new policy, should be one at the first iteration
    ratio = torch.exp(log_prob - rollout_data.old_log_prob)

    # clipped surrogate loss
    policy_loss_1 = advantages * ratio
    policy_loss_2 = advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
    policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

    if agent.clip_range_vf is None:
        # No clipping
        values_pred = values
    else:
        # Clip the different between old and new value
        # NOTE: this depends on the reward scaling
        values_pred = rollout_data.old_values + torch.clamp(
            values - rollout_data.old_values, -clip_range_vf, clip_range_vf
        )
    # Value loss using the TD(gae_lambda) target
    value_loss = torch.nn.functional.mse_loss(rollout_data.returns, values_pred)

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
        ratio.mean(),
    )
