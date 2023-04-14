from typing import Optional, Union

import gym
import numpy as np
import pytest
import stable_baselines3
from gym.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from action_space_toolbox.analysis.util import evaluate_agent_returns
from action_space_toolbox.util.agent_spec import AgentSpec
from action_space_toolbox.util.angles import normalize_angle
from action_space_toolbox.util.sb3_training import fill_rollout_buffer
from action_space_toolbox.util.tensorboard_logs import merge_dicts


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


class DummyEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.counter = 0
        self.max_step = 50
        self.observation_space = gym.spaces.Box(
            np.zeros(1, dtype=np.float32), np.ones(1, dtype=np.float32)
        )
        self.action_space = gym.spaces.Box(
            -np.ones(1, dtype=np.float32), np.ones(1, dtype=np.float32)
        )

    def step(self, action):
        self.counter += 1
        return (
            np.array([self.counter / self.max_step]),
            float(self.counter),
            False,
            {},
        )

    def reset(self):
        self.counter = 0
        return np.zeros(1)

    def render(self, mode="human"):
        return


def env_factory():
    return TimeLimit(DummyEnv(), 5)


class DummyAgentSpec(AgentSpec):
    def __init__(self, env):
        super().__init__(None, None)
        self.env = env

    def _create_agent(
        self,
        env: Optional[Union[gym.Env, stable_baselines3.common.vec_env.VecEnv]] = None,
    ) -> stable_baselines3.ppo.PPO:
        return PPO("MlpPolicy", DummyVecEnv([lambda: self.env]), device="cpu", seed=42)


def test_fill_rollout_buffer():
    for num_steps in [300, 303]:
        env = env_factory()
        rollout_buffer = RolloutBuffer(
            num_steps, env.observation_space, env.action_space, device="cpu"
        )
        agent_spec = DummyAgentSpec(env)
        fill_rollout_buffer(
            env_factory, agent_spec, rollout_buffer, num_spawned_processes=3
        )

        rollout_buffer_ppo = RolloutBuffer(
            num_steps, env.observation_space, env.action_space, device="cpu"
        )
        ppo = agent_spec.create_agent(env_factory())
        ppo._last_obs = ppo.env.reset()
        ppo._last_episode_starts = True
        callback = EvalCallback(DummyVecEnv([env_factory]))
        callback.init_callback(ppo)
        ppo.collect_rollouts(ppo.env, callback, rollout_buffer_ppo, num_steps)
        assert rollout_buffer.observations == pytest.approx(
            rollout_buffer_ppo.observations
        )
        assert rollout_buffer.rewards == pytest.approx(rollout_buffer_ppo.rewards)
        assert rollout_buffer.episode_starts == pytest.approx(
            rollout_buffer_ppo.episode_starts
        )
        assert rollout_buffer.values == pytest.approx(rollout_buffer_ppo.values)
        assert rollout_buffer.advantages == pytest.approx(rollout_buffer_ppo.advantages)
        assert rollout_buffer.returns == pytest.approx(rollout_buffer_ppo.returns)


def test_merge_dicts():
    d1 = {1: 2, "sub1": {2: 3}, "sub2": {4: 5}}
    d2 = {3: 4, "sub1": {6: 7}, "sub3": {23: 42}}

    expected = {1: 2, 3: 4, "sub1": {2: 3, 6: 7}, "sub2": {4: 5}, "sub3": {23: 42}}
    assert merge_dicts(d1, d2) == expected


def test_evaluate_agent_returns():
    env = gym.wrappers.TimeLimit(DummyEnv(), 50)
    agent = PPO("MlpPolicy", DummyVecEnv([lambda: env]), device="cpu", seed=42)

    eval_result_steps = evaluate_agent_returns(agent, env, num_steps=123)
    eval_result_episodes = evaluate_agent_returns(agent, env, num_episodes=2)
    gt_value_undiscounted = np.sum(np.arange(1, env.max_step + 1))
    gt_value_discounted = np.sum(
        agent.gamma ** np.arange(env.max_step) * np.arange(1, env.max_step + 1)
    )

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
