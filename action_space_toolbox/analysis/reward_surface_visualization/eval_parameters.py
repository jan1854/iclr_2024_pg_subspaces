from typing import Sequence, Callable

import gym
import stable_baselines3.common.base_class
import torch
import numpy as np
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.vec_env import DummyVecEnv


def sample_episode_rewards(
    env: stable_baselines3.common.vec_env.VecEnv,
    agent: stable_baselines3.common.base_class.BaseAlgorithm,
    num_steps: int,
) -> np.ndarray:
    obs = env.reset()
    episode_rewards = []
    reward_curr_episode = 0.0
    for _ in range(num_steps):
        action, policy_val, policy_log_prob = agent.policy(
            obs_as_tensor(obs, device=agent.device)
        )
        action = action.detach().cpu().numpy()
        action = np.clip(action, agent.action_space.low, agent.action_space.high)
        obs, reward, done, _ = env.step(action)
        reward_curr_episode += reward
        if done:
            obs = env.reset()
            episode_rewards.append(reward_curr_episode)
            reward_curr_episode = 0.0
    return np.array(episode_rewards)


def eval_parameters(
    agent_weights: Sequence[torch.Tensor],
    env_factory: Callable[[], gym.Env],
    agent_factory: Callable[[], stable_baselines3.common.base_class.BaseAlgorithm],
    num_steps: int,
) -> float:
    env = DummyVecEnv([env_factory])
    agent = agent_factory()
    with torch.no_grad():
        for parameters, weights in zip(agent.policy.parameters(), agent_weights):
            parameters.data[:] = weights

    episode_rewards = sample_episode_rewards(env, agent, num_steps)
    return episode_rewards.mean()
