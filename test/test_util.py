import gym
import numpy as np
import pytest
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from action_space_toolbox.analysis.util import (
    evaluate_agent_returns,
)
from action_space_toolbox.util.angles import normalize_angle
from action_space_toolbox.util.tensorboard_logs import merge_dicts


class DummyEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.counter = 0
        self.max_steps = [23, 50, 12, 9, 15]
        self.observation_space = gym.spaces.Box(
            np.zeros(1, dtype=np.float32), np.ones(1, dtype=np.float32)
        )
        self.action_space = gym.spaces.Box(
            -np.ones(1, dtype=np.float32), np.ones(1, dtype=np.float32)
        )
        self.curr_episode = -1

    def step(self, action):
        self.counter += 1
        done = self.counter == self.max_steps[self.curr_episode % len(self.max_steps)]
        return (
            np.array([self.counter / 50]),
            float(self.counter),
            done,
            {},
        )


def test_angle_normalization():
    assert normalize_angle(np.pi) == pytest.approx(np.pi)
    assert normalize_angle(-np.pi) == pytest.approx(np.pi)
    assert normalize_angle(0.0) == pytest.approx(0.0)
    assert normalize_angle(np.pi + 0.05) == pytest.approx(-np.pi + 0.05)
    assert normalize_angle(-np.pi - 0.05) == pytest.approx(np.pi - 0.05)
    assert normalize_angle(4 * np.pi) == pytest.approx(0.0)
    assert normalize_angle(3 * np.pi) == pytest.approx(np.pi)
    assert normalize_angle(
        np.array([0.5 * np.pi, 1.2 * np.pi, -1.3 * np.pi])
    ) == pytest.approx(np.array([0.5 * np.pi, -0.8 * np.pi, 0.7 * np.pi]))


def test_merge_dicts():
    d1 = {1: 2, "sub1": {2: 3}, "sub2": {4: 5}}
    d2 = {3: 4, "sub1": {6: 7}, "sub3": {23: 42}}

    expected = {1: 2, 3: 4, "sub1": {2: 3, 6: 7}, "sub2": {4: 5}, "sub3": {23: 42}}
    assert merge_dicts(d1, d2) == expected


def test_evaluate_agent_returns():
    env = gym.wrappers.TimeLimit(DummyEnv(), 50)
    agent = PPO("MlpPolicy", DummyVecEnv([lambda: env]), device="cpu", seed=42)

    num_episodes = 4
    num_steps = sum(env.max_steps[:num_episodes]) + 3
    env = gym.wrappers.TimeLimit(DummyEnv(), 50)
    eval_result_steps = evaluate_agent_returns(agent, env, num_steps=num_steps)
    env = gym.wrappers.TimeLimit(DummyEnv(), 50)
    eval_result_episodes = evaluate_agent_returns(agent, env, num_episodes=num_episodes)
    gt_values_undiscounted = []
    gt_values_discounted = []
    for episode in range(num_episodes):
        curr_max_step = env.max_steps[episode]
        gt_values_undiscounted.append(np.sum(np.arange(1, curr_max_step + 1)))
        gt_values_discounted.append(
            np.sum(
                agent.gamma ** np.arange(curr_max_step)
                * np.arange(1, curr_max_step + 1)
            )
        )
    gt_value_undiscounted = np.mean(gt_values_undiscounted)
    gt_value_discounted = np.mean(gt_values_discounted)

    assert eval_result_steps.rewards_undiscounted.item() == pytest.approx(
        gt_value_undiscounted
    )
    assert eval_result_episodes.rewards_undiscounted.item() == pytest.approx(
        gt_value_undiscounted
    )
    assert eval_result_steps.rewards_discounted.item() == pytest.approx(
        gt_value_discounted
    )
    assert eval_result_episodes.rewards_discounted.item() == pytest.approx(
        gt_value_discounted
    )
