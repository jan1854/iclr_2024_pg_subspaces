import re
import tempfile
from pathlib import Path

import gym
import numpy as np
import pytest
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from pg_subspaces.analysis.util import (
    evaluate_agent_returns,
)
from pg_subspaces.callbacks.custom_checkpoint_callback import CustomCheckpointCallback
from pg_subspaces.metrics.tensorboard_logs import merge_dicts
from pg_subspaces.sb3_utils.common.replay_buffer_diff_checkpointer import (
    ReplayBufferDiffCheckpointer,
)


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

    def reset(self):
        self.counter = 0
        self.curr_episode += 1
        return np.zeros(1)


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
    env = DummyVecEnv([lambda: gym.wrappers.TimeLimit(DummyEnv(), 50)])
    eval_result_steps = evaluate_agent_returns(agent, env, num_steps=num_steps)
    env = DummyVecEnv([lambda: gym.wrappers.TimeLimit(DummyEnv(), 50)])
    eval_result_episodes = evaluate_agent_returns(agent, env, num_episodes=num_episodes)
    gt_values_undiscounted = []
    gt_values_discounted = []
    for episode in range(num_episodes):
        curr_max_step = env.envs[0].max_steps[episode]
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


def test_replay_buffer_checkpointing():
    env = DummyVecEnv([lambda: gym.make("Pendulum-v1")])
    algo = SAC(
        "MlpPolicy",
        env,
        buffer_size=20,
        policy_kwargs={"net_arch": [32, 32]},
        device="cpu",
    )
    with tempfile.TemporaryDirectory() as tempdir:
        tempdir = Path(tempdir)
        callbacks = [
            CustomCheckpointCallback(19, [], tempdir, "sac", True),
            CheckpointCallback(19, tempdir, "sac", True),
        ]
        algo.learn(500, callbacks)

        for checkpoint in tempdir.glob("sac_[0-9]*_steps.zip"):
            step = int(re.search("_[0-9]+_", checkpoint.name).group()[1:-1])
            algo = SAC.load(checkpoint)
            algo.load_replay_buffer(tempdir / f"sac_replay_buffer_{step}_steps.pkl")
            replay_buffer_checkpointer = ReplayBufferDiffCheckpointer(
                algo, "sac", tempdir
            )

            # The replay buffer loaded with the regular sb3 checkpointing
            obs_complete = algo.replay_buffer.observations.copy()
            act_complete = algo.replay_buffer.actions.copy()
            next_obs_complete = algo.replay_buffer.next_observations.copy()
            reward_complete = algo.replay_buffer.rewards.copy()
            done_complete = algo.replay_buffer.dones.copy()
            pos_complete = algo.replay_buffer.pos
            full_complete = algo.replay_buffer.full

            # Make sure that the replay buffer is modified
            algo.replay_buffer.observations = np.zeros_like(
                algo.replay_buffer.observations
            )
            algo.replay_buffer.actions = np.zeros_like(algo.replay_buffer.actions)
            algo.replay_buffer.next_observations = np.zeros_like(
                algo.replay_buffer.next_observations
            )
            algo.replay_buffer.rewards = np.zeros_like(algo.replay_buffer.rewards)
            algo.replay_buffer.dones = np.zeros_like(algo.replay_buffer.dones)
            algo.replay_buffer.pos = -1
            algo.replay_buffer.full = None

            replay_buffer_checkpointer.load(step)
            assert np.all(algo.replay_buffer.observations == obs_complete)
            assert np.all(algo.replay_buffer.actions == act_complete)
            assert np.all(algo.replay_buffer.next_observations == next_obs_complete)
            assert np.all(algo.replay_buffer.rewards == reward_complete)
            assert np.all(algo.replay_buffer.dones == done_complete)
            assert algo.replay_buffer.pos == pos_complete
            assert algo.replay_buffer.full == full_complete
