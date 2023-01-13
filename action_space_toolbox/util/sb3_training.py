from typing import Optional, Tuple

import gym
import numpy as np
import stable_baselines3.common.vec_env
import stable_baselines3.common.buffers
import torch
import torch.nn.functional as F
from stable_baselines3.common.utils import obs_as_tensor
from tqdm import tqdm


def fill_rollout_buffer(
    agent: stable_baselines3.ppo.PPO,
    env: stable_baselines3.common.vec_env.VecEnv,
    rollout_buffer: Optional[stable_baselines3.common.buffers.RolloutBuffer],
    rollout_buffer_no_value_bootstrap: Optional[
        stable_baselines3.common.buffers.RolloutBuffer
    ] = None,
    show_progress: bool = False,
) -> None:
    """
    Collect experiences using the current policy and fill a ``RolloutBuffer``. The code is adapted from
    stable-baselines3's OnPolicyAlgorithm.collect_rollouts() (we cannot use that function since it modifies the
    state of the agent (e.g. the number of timesteps)).

    :param agent:                               The RL agent
    :param env:                                 The training environment
    :param rollout_buffer:                      Buffer to fill with rollouts
    :param rollout_buffer_no_value_bootstrap:   A separate buffer for without the value bootstrap (i.e, the last reward
                                                of truncated episodes is not modified to reward + gamma * next_value)
    :param show_progress:                       Shows a progress bar if true
    """
    assert rollout_buffer is not None or rollout_buffer_no_value_bootstrap is not None
    assert (
        rollout_buffer_no_value_bootstrap is None
        or rollout_buffer is None
        or rollout_buffer.buffer_size == rollout_buffer_no_value_bootstrap.buffer_size
    )
    buffer_size = (
        rollout_buffer.buffer_size
        if rollout_buffer is not None
        else rollout_buffer_no_value_bootstrap.buffer_size
    )

    last_obs = env.reset()
    last_episode_starts = np.ones(env.num_envs)

    # Switch to eval mode (this affects batch norm / dropout)
    agent.policy.set_training_mode(False)

    if rollout_buffer is not None:
        rollout_buffer.reset()
    if rollout_buffer_no_value_bootstrap is not None:
        rollout_buffer_no_value_bootstrap.reset()
    # Sample new weights for the state dependent exploration
    if agent.use_sde:
        agent.policy.reset_noise(env.num_envs)

    for n_steps in tqdm(
        range(buffer_size),
        disable=not show_progress,
        mininterval=300,
        desc="Collecting samples",
        unit="samples",
    ):
        if (
            agent.use_sde
            and agent.sde_sample_freq > 0
            and (n_steps - 1) % agent.sde_sample_freq == 0
        ):
            # Sample a new noise matrix
            agent.policy.reset_noise(env.num_envs)

        with torch.no_grad():
            # Convert to pytorch tensor or to TensorDict
            obs_tensor = obs_as_tensor(last_obs, agent.device)
            actions, values, log_probs = agent.policy(obs_tensor)
        actions = actions.cpu().numpy()

        # Rescale and perform action
        clipped_actions = actions
        # Clip the actions to avoid out of bounds error
        if isinstance(agent.action_space, gym.spaces.Box):
            clipped_actions = np.clip(
                actions, agent.action_space.low, agent.action_space.high
            )

        new_obs, rewards, dones, infos = env.step(clipped_actions)

        if isinstance(agent.action_space, gym.spaces.Discrete):
            # Reshape in case of discrete action
            actions = actions.reshape(-1, 1)

        # Handle timeout by bootstrapping with value function
        # see GitHub issue #633
        rewards_no_bootstrap = rewards.copy()
        for idx, done in enumerate(dones):
            if (
                done
                and infos[idx].get("terminal_observation") is not None
                and infos[idx].get("TimeLimit.truncated", False)
            ):
                terminal_obs = agent.policy.obs_to_tensor(
                    infos[idx]["terminal_observation"]
                )[0]
                with torch.no_grad():
                    terminal_value = agent.policy.predict_values(terminal_obs)[0]
                rewards[idx] += agent.gamma * terminal_value

        if rollout_buffer is not None:
            rollout_buffer.add(
                last_obs,
                actions,
                rewards,
                last_episode_starts,
                values,
                log_probs,
            )
        if rollout_buffer_no_value_bootstrap is not None:
            rollout_buffer_no_value_bootstrap.add(
                last_obs,
                actions,
                rewards_no_bootstrap,
                last_episode_starts,
                values,
                log_probs,
            )

        last_obs = new_obs
        last_episode_starts = dones

    with torch.no_grad():
        # Compute value for the last timestep
        values = agent.policy.predict_values(obs_as_tensor(new_obs, agent.device))
    if rollout_buffer is not None:
        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)
    if rollout_buffer_no_value_bootstrap is not None:
        rollout_buffer_no_value_bootstrap.compute_returns_and_advantage(
            last_values=values, dones=dones
        )


def ppo_loss(
    agent: stable_baselines3.ppo.PPO,
    rollout_data: stable_baselines3.common.buffers.RolloutBufferSamples,
) -> Tuple[torch.Tensor, torch.Tensor]:
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
    value_loss = F.mse_loss(rollout_data.returns, values_pred)

    # Entropy loss favor exploration
    if entropy is None:
        # Approximate entropy when no analytical form
        entropy_loss = -torch.mean(-log_prob)
    else:
        entropy_loss = -torch.mean(entropy)

    loss = policy_loss + agent.ent_coef * entropy_loss + agent.vf_coef * value_loss
    return loss, ratio.mean()
