import collections
import functools
import math
from typing import Optional, Tuple, List, Callable

import gym
import numpy as np
import stable_baselines3.common.vec_env
import stable_baselines3.common.buffers
import torch
import torch.nn.functional as F
from stable_baselines3.common.type_aliases import RolloutBufferSamples
from stable_baselines3.common.utils import obs_as_tensor

from action_space_toolbox.util.get_episode_length import get_episode_length

EnvSteps = collections.namedtuple(
    "EnvSteps",
    "observations actions rewards rewards_no_bootstrap dones values log_probs",
)


def fill_rollout_buffer(
    env_factory: Callable[[], gym.Env],
    agent_factory: Callable[[gym.Env], stable_baselines3.ppo.PPO],
    rollout_buffer: Optional[stable_baselines3.common.buffers.RolloutBuffer],
    rollout_buffer_no_value_bootstrap: Optional[
        stable_baselines3.common.buffers.RolloutBuffer
    ] = None,
    num_processes: int = 1,
) -> None:
    """
    Collect experiences using the current policy and fill a ``RolloutBuffer``. The code is adapted from
    stable-baselines3's OnPolicyAlgorithm.collect_rollouts() (we cannot use that function since it modifies the
    state of the agent (e.g. the number of timesteps)).

    :param agent_factory:                       A function that creates the RL agent
    :param env_factory:                         A function that creates the environment
    :param rollout_buffer:                      Buffer to fill with rollouts
    :param rollout_buffer_no_value_bootstrap:   A separate buffer for without the value bootstrap (i.e, the last reward
                                                of truncated episodes is not modified to reward + gamma * next_value)
    :param num_processes:                       The number of processes to use
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

    if num_processes == 1:
        env_steps = [collect_complete_episodes(buffer_size, env_factory, agent_factory)]
    else:
        jobs = [math.ceil(buffer_size / num_processes)] * (num_processes - 1) + [
            buffer_size - math.ceil(buffer_size / num_processes) * (num_processes - 1)
        ]
        with torch.multiprocessing.get_context("spawn").Pool(num_processes) as pool:
            env_steps = pool.map(
                functools.partial(
                    collect_complete_episodes,
                    env_factory=env_factory,
                    agent_factory=agent_factory,
                ),
                jobs,
            )

    if rollout_buffer is not None:
        rollout_buffer.reset()
    if rollout_buffer_no_value_bootstrap is not None:
        rollout_buffer_no_value_bootstrap.reset()

    final_next_obs = None
    final_done = None
    for curr_env_steps in env_steps:
        elements_to_add = min(
            rollout_buffer.buffer_size - rollout_buffer.pos, len(curr_env_steps.actions)
        )
        if elements_to_add == 0:
            break
        episode_starts = np.concatenate(([True], curr_env_steps.dones[:-1]))
        rbs = []
        rews = []
        if rollout_buffer is not None:
            rbs.append(rollout_buffer)
            rews.append(curr_env_steps.rewards)
        if rollout_buffer_no_value_bootstrap is not None:
            rbs.append(rollout_buffer_no_value_bootstrap)
            rews.append(curr_env_steps.rewards_no_value_bootstrap)
        for rb, rew in zip(rbs, rews):
            # This is not how stable-baselines3's RolloutBuffer is intended to be used, but this is way faster than
            # adding each element individually.
            next_pos = rb.pos + elements_to_add
            rb.observations[rb.pos : next_pos, 0, :] = curr_env_steps.observations[
                :elements_to_add
            ]
            rb.actions[rb.pos : next_pos, 0, :] = curr_env_steps.actions[
                :elements_to_add
            ]
            rb.rewards[rb.pos : next_pos, 0] = rew[:elements_to_add]
            rb.episode_starts[rb.pos : next_pos, 0] = episode_starts[:elements_to_add]
            rb.values[rb.pos : next_pos, 0] = curr_env_steps.values[:elements_to_add]
            rb.log_probs[rb.pos : next_pos, 0] = curr_env_steps.log_probs[
                :elements_to_add
            ]
            rb.pos = next_pos
        final_done = curr_env_steps.dones[elements_to_add - 1]
        if elements_to_add < len(curr_env_steps.observations):
            final_next_obs = curr_env_steps.observations[elements_to_add]

    if rollout_buffer is not None:
        rollout_buffer.full = True
    if rollout_buffer_no_value_bootstrap is not None:
        rollout_buffer_no_value_bootstrap.full = True

    del env_steps

    agent = agent_factory(env_factory())
    if final_next_obs is None:
        # In the case that we sampled exactly the required number of steps, the last step will be the end of an episode
        # (because we only sample full episodes). Therefore, the value argument is irrelevant (as it will be multiplied
        # by zero in RolloutBuffer.compute_returns_and_advantage() anyway).
        value = torch.zeros((1, 1), device=agent.device)
    else:
        with torch.no_grad():
            # Compute value for the last timestep
            value = agent.policy.predict_values(
                obs_as_tensor(final_next_obs, agent.device).unsqueeze(0)
            )
    if rollout_buffer is not None:
        rollout_buffer.compute_returns_and_advantage(
            last_values=value, dones=np.array([[final_done]])
        )
    if rollout_buffer_no_value_bootstrap is not None:
        rollout_buffer_no_value_bootstrap.compute_returns_and_advantage(
            last_values=value, dones=np.array([[final_done]])
        )


def collect_complete_episodes(
    min_num_env_steps: int,
    env_factory: Callable[[], gym.Env],
    agent_factory: Callable[[gym.Env], stable_baselines3.ppo.PPO],
) -> EnvSteps:
    env = env_factory()
    agent = agent_factory(env)

    episode_length = get_episode_length(env)
    arr_len = min_num_env_steps + episode_length + 1
    observations = np.empty((arr_len, *env.observation_space.shape), dtype=np.float32)
    actions = np.empty((arr_len, *env.action_space.shape), dtype=np.float32)
    rewards = np.empty(arr_len, dtype=np.float32)
    rewards_no_bootstrap = np.empty(arr_len, dtype=np.float32)
    dones = np.empty(arr_len, dtype=bool)
    values = np.empty(arr_len, dtype=np.float32)
    log_probs = np.empty(arr_len, dtype=np.float32)

    observations[0] = env.reset()

    # Switch to eval mode (this affects batch norm / dropout)
    agent.policy.set_training_mode(False)

    # Sample new weights for the state dependent exploration
    if agent.use_sde:
        agent.policy.reset_noise()

    n_steps = 0
    done = False
    while n_steps < min_num_env_steps or not done:
        if (
            agent.use_sde
            and agent.sde_sample_freq > 0
            and (n_steps - 1) % agent.sde_sample_freq == 0
        ):
            # Sample a new noise matrix
            agent.policy.reset_noise()

        with torch.no_grad():
            # Convert to pytorch tensor or to TensorDict
            obs_tensor = obs_as_tensor(observations[n_steps], agent.device)
            action, value, log_prob = agent.policy(obs_tensor.unsqueeze(0))
        action = action.squeeze(0).cpu().numpy()

        # Rescale and perform action
        clipped_action = action
        # Clip the action to avoid out of bounds error
        if isinstance(agent.action_space, gym.spaces.Box):
            clipped_action = np.clip(
                action, agent.action_space.low, agent.action_space.high
            )

        obs, reward, done, info = env.step(clipped_action)

        if isinstance(agent.action_space, gym.spaces.Discrete):
            # Reshape in case of discrete action
            action = action.reshape(-1, 1)

        # Handle timeout by bootstrapping with value function
        # see GitHub issue #633
        reward_no_bootstrap = reward
        if done and info.get("TimeLimit.truncated", False):
            terminal_obs = agent.policy.obs_to_tensor(obs)[0]
            with torch.no_grad():
                terminal_value = agent.policy.predict_values(terminal_obs)[0]
            reward += agent.gamma * terminal_value.item()

        if done:
            obs = env.reset()

        # This is for the next step (we add the first observation before the loop); the last observation of each episode
        # is not added to observations since it is not needed for filling a RolloutBuffer.
        observations[n_steps + 1, :] = obs
        actions[n_steps, :] = action
        rewards[n_steps] = reward
        rewards_no_bootstrap[n_steps] = reward_no_bootstrap
        dones[n_steps] = done
        values[n_steps] = value.item()
        log_probs[n_steps] = log_prob.item()

        n_steps += 1

    # Check that the last episode is complete
    assert dones[n_steps - 1]

    return EnvSteps(
        observations[:n_steps],
        actions[:n_steps],
        rewards[:n_steps],
        rewards_no_bootstrap[:n_steps],
        dones[:n_steps],
        values[:n_steps],
        log_probs[:n_steps],
    )


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
    value_loss = F.mse_loss(rollout_data.returns, values_pred)

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


def ppo_gradient(
    agent: stable_baselines3.ppo.PPO, rollout_data: RolloutBufferSamples
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    loss, _, _, _ = ppo_loss(agent, rollout_data)
    agent.policy.zero_grad()
    loss.backward()
    combined_gradient = [p.grad for p in agent.policy.parameters()]
    policy_gradient = [p.grad for p in get_policy_parameters(agent)]
    value_function_gradient = [p.grad for p in get_value_function_parameters(agent)]
    return combined_gradient, policy_gradient, value_function_gradient


def get_policy_parameters(agent: stable_baselines3.ppo.PPO) -> List[torch.nn.Parameter]:
    policy = agent.policy
    assert policy.share_features_extractor
    params_feature_extractor = list(policy.features_extractor.parameters()) + list(
        policy.pi_features_extractor.parameters()
    )
    params_mlp_extractor = list(policy.mlp_extractor.shared_net.parameters()) + list(
        policy.mlp_extractor.policy_net.parameters()
    )
    params_action_net = list(policy.action_net.parameters())
    return params_feature_extractor + params_mlp_extractor + params_action_net


def get_value_function_parameters(
    agent: stable_baselines3.ppo.PPO,
) -> List[torch.nn.Parameter]:
    policy = agent.policy
    assert policy.share_features_extractor
    params_feature_extractor = list(policy.features_extractor.parameters()) + list(
        policy.vf_features_extractor.parameters()
    )
    params_mlp_extractor = list(policy.mlp_extractor.shared_net.parameters()) + list(
        policy.mlp_extractor.value_net.parameters()
    )
    params_value_net = list(policy.value_net.parameters())
    return params_feature_extractor + params_mlp_extractor + params_value_net
