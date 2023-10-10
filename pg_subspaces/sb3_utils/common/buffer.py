import collections
import functools
import math
import re
from pathlib import Path
from typing import Callable, List, Optional, Union, Sequence, Tuple

import gym
import numpy as np
import stable_baselines3
import stable_baselines3.common.buffers
import stable_baselines3.common.off_policy_algorithm
import stable_baselines3.common.vec_env
import torch
from stable_baselines3.common.type_aliases import (
    ReplayBufferSamples,
    RolloutBufferSamples,
)
from stable_baselines3.common.utils import obs_as_tensor

from pg_subspaces.sb3_utils.common.agent_spec import AgentSpec
from pg_subspaces.sb3_utils.common.training import (
    maybe_create_agent,
    maybe_create_env,
    check_wrapped,
)

EnvSteps = collections.namedtuple(
    "EnvSteps",
    "observations actions rewards rewards_no_value_bootstrap dones values log_probs",
)


def fill_rollout_buffer(
    env_or_factory: Union[gym.Env, Callable[[], gym.Env]],
    agent_or_spec: Union[AgentSpec, stable_baselines3.ppo.PPO],
    rollout_buffer: Optional[stable_baselines3.common.buffers.RolloutBuffer],
    rollout_buffer_no_value_bootstrap: Optional[
        stable_baselines3.common.buffers.RolloutBuffer
    ] = None,
    max_num_episodes: Optional[int] = None,
    num_spawned_processes: int = 0,
) -> bool:
    """
    Collect experiences using the current policy and fill a ``RolloutBuffer``. The code is adapted from
    stable-baselines3's OnPolicyAlgorithm.collect_rollouts() (we cannot use that function since it modifies the
    state of the agent (e.g. the number of timesteps)).

    :param agent_or_spec:                       An AgentSpec or a PPO agent
    :param env_or_factory:                      An environment or a function that creates an environment
    :param rollout_buffer:                      Buffer to fill with rollouts
    :param rollout_buffer_no_value_bootstrap:   A separate buffer for without the value bootstrap (i.e, the last reward
                                                of truncated episodes is not modified to reward + gamma * next_value)
    :param num_spawned_processes:               The number of processes to spawn for collecting the samples (if 0, the
                                                current process will be used)
    :return:                                    Whether the last episode is complete (necessary since the rollout buffer
                                                only stores the episode starts)
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

    assert num_spawned_processes >= 0
    if num_spawned_processes == 0:
        env_steps = [
            collect_complete_episodes(
                buffer_size, max_num_episodes, env_or_factory, agent_or_spec
            )
        ]
    else:
        # TODO: Clean this up
        jobs = [
            (
                math.ceil(buffer_size / num_spawned_processes),
                math.ceil(max_num_episodes / num_spawned_processes)
                if max_num_episodes is not None
                else None,
            )
        ] * (num_spawned_processes - 1) + [
            (
                buffer_size
                - math.ceil(buffer_size / num_spawned_processes)
                * (num_spawned_processes - 1),
                (
                    max_num_episodes
                    - math.ceil(max_num_episodes / num_spawned_processes)
                    * (num_spawned_processes - 1)
                )
                if max_num_episodes is not None
                else None,
            )
        ]
        with torch.multiprocessing.get_context("spawn").Pool(
            num_spawned_processes
        ) as pool:
            env_steps = pool.starmap(
                functools.partial(
                    collect_complete_episodes,
                    env_or_factory=env_or_factory,
                    agent_or_spec=agent_or_spec,
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
        episode_starts = np.concatenate(([True], curr_env_steps.dones[:-1]))
        rbs = []
        rews = []
        if rollout_buffer is not None:
            rbs.append(rollout_buffer)
            rews.append(curr_env_steps.rewards)
        if rollout_buffer_no_value_bootstrap is not None:
            rbs.append(rollout_buffer_no_value_bootstrap)
            rews.append(curr_env_steps.rewards_no_value_bootstrap)

        elements_to_add = min(
            rbs[0].buffer_size - rbs[0].pos, len(curr_env_steps.actions)
        )
        if elements_to_add == 0:
            break
        for rb, rew in zip(rbs, rews):
            # This is not how stable-baselines3's RolloutBuffer is intended to be used, but this is way faster than
            # adding each element individually.
            next_pos = rb.pos + elements_to_add
            # Reshape is needed to handle discrete action spaces
            rb.observations[rb.pos : next_pos, 0, :] = curr_env_steps.observations[
                :elements_to_add
            ].reshape((elements_to_add,) + rb.obs_shape)
            rb.actions[rb.pos : next_pos, 0, :] = curr_env_steps.actions[
                :elements_to_add
            ].reshape(elements_to_add, rb.action_dim)
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

    agent = maybe_create_agent(agent_or_spec, maybe_create_env(env_or_factory))
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
    return final_done


def get_episode_length(
    env: Union[gym.Env, stable_baselines3.common.vec_env.VecEnv]
) -> int:
    # This is quite an ugly hack but there is no elegant way to get the environment's time limit at the moment
    # TODO: This does not work with SubprocVecEnvs
    if isinstance(env, stable_baselines3.common.vec_env.VecEnv):
        assert len(env.envs) == 1
        env = env.envs[0]
    assert check_wrapped(env, gym.wrappers.TimeLimit)
    while not hasattr(env, "_max_episode_steps"):
        env = env.env
    return env._max_episode_steps


def collect_complete_episodes(
    min_num_env_steps: int,
    max_num_episodes: Optional[int],
    env_or_factory: Union[gym.Env, Callable[[], gym.Env]],
    agent_or_spec: Union[AgentSpec, stable_baselines3.ppo.PPO],
) -> EnvSteps:
    if isinstance(env_or_factory, Callable):
        env = env_or_factory()
    else:
        env = env_or_factory
    agent = maybe_create_agent(agent_or_spec, env)

    episode_length = get_episode_length(env)
    arr_len = min_num_env_steps + episode_length + 1
    observations = np.empty(
        (arr_len, *env.observation_space.shape),
        dtype=env.observation_space.dtype,
    )
    actions = np.empty((arr_len, *env.action_space.shape), dtype=env.action_space.dtype)
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
    n_episodes = 0
    done = False
    while (n_steps < min_num_env_steps or not done) and (
        max_num_episodes is None or n_episodes < max_num_episodes
    ):
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
            n_episodes += 1

        # This is for the next step (we add the first observation before the loop); the last observation of each episode
        # is not added to observations since it is not needed for filling a RolloutBuffer.
        observations[n_steps + 1] = obs
        actions[n_steps] = action
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


def concatenate_buffer_samples(
    batches: Union[Sequence[ReplayBufferSamples], Sequence[RolloutBufferSamples]]
) -> Union[ReplayBufferSamples, RolloutBufferSamples]:
    if isinstance(batches[0], ReplayBufferSamples):
        return ReplayBufferSamples(
            *(
                torch.cat([getattr(data, field) for data in batches])
                for field in ReplayBufferSamples._fields
            )
        )
    else:
        return RolloutBufferSamples(
            *(
                torch.cat([getattr(data, field) for data in batches])
                for field in RolloutBufferSamples._fields
            )
        )
