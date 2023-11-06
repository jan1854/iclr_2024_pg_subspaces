import itertools
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union

import gym
import numpy as np
import pytest
import stable_baselines3
import torch
from gym.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from pg_subspaces.callbacks.custom_checkpoint_callback import CustomCheckpointCallback
from pg_subspaces.sb3_utils.common.agent_spec import AgentSpec
from pg_subspaces.sb3_utils.common.buffer import fill_rollout_buffer
from pg_subspaces.sb3_utils.common.replay_buffer_diff_checkpointer import (
    ReplayBufferDiffCheckpointer,
)
from pg_subspaces.sb3_utils.common.loss import actor_critic_gradient
from pg_subspaces.sb3_utils.common.parameters import (
    get_actor_critic_parameters,
    flatten_parameters,
    unflatten_parameters_for_agent,
    project,
    project_orthonormal,
    project_orthonormal_inverse,
    combine_actor_critic_parameter_vectors,
    get_trained_parameters,
)


def tensor_in(a: torch.Tensor, seq: Sequence[torch.Tensor]) -> bool:
    for b in seq:
        if a.shape == b.shape and torch.all(a == b):
            return True
    return False


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

    def render(self, mode="human"):
        return


def env_factory():
    return TimeLimit(DummyEnv(), 5)


class DummyAgentSpec(AgentSpec):
    def __init__(self, env):
        super().__init__("cpu", None, None)
        self.env = env

    def _create_agent(
        self,
        env: Optional[Union[gym.Env, stable_baselines3.common.vec_env.VecEnv]] = None,
    ) -> stable_baselines3.ppo.PPO:
        return PPO("MlpPolicy", DummyVecEnv([lambda: self.env]), device="cpu", seed=42)

    def copy_with_new_parameters(
        self,
        weights: Optional[Sequence[torch.Tensor]] = None,
        agent_kwargs: Optional[Dict[str, Any]] = None,
    ) -> "DummyAgentSpec":
        return DummyAgentSpec(self.env)


def test_get_parameters():
    agent = stable_baselines3.ppo.PPO(
        "MlpPolicy", gym.make("Pendulum-v1"), device="cpu"
    )
    policy_parameters, vf_parameters = get_actor_critic_parameters(agent)
    # To ensure that the parameters are unique
    for p in agent.policy.parameters():
        p.requires_grad = False
        p[:] = torch.normal(0, 1, size=p.shape)
    for p in agent.policy.parameters():
        assert tensor_in(p, policy_parameters) != tensor_in(p, vf_parameters)
    for p in itertools.chain(policy_parameters, vf_parameters):
        assert tensor_in(p, agent.policy.parameters())


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


def test_projection():
    vec = torch.tensor([[1.0, 1.0, 1.0]]).T
    subspace = torch.tensor([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]).T
    vec_subspace = project(vec, subspace, result_in_orig_space=False)
    assert vec_subspace == pytest.approx(torch.tensor([[1.0, 0.5]]).T)
    vec_orig_space = project(vec, subspace, result_in_orig_space=True)
    assert vec_orig_space == pytest.approx(torch.tensor([[1.0, 1.0, 0.0]]).T)

    vecs = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]]).T
    vecs_subspace = project(vecs, subspace, result_in_orig_space=False)
    assert vecs_subspace == pytest.approx(torch.tensor([[1.0, 0.0], [0.0, 0.5]]).T)
    vecs_orig_space = project(vecs, subspace, result_in_orig_space=True)
    assert vecs_orig_space == pytest.approx(
        torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]).T
    )


def test_project_inverse():
    random_vectors = torch.randn(100, 50)
    subspace, _ = torch.linalg.qr(random_vectors)
    vec_subspace = torch.randn(50, 1)
    vec_orig_space = project_orthonormal_inverse(vec_subspace, subspace)
    vec_backprojected = project(vec_orig_space, subspace, result_in_orig_space=False)
    vec_backprojected_orthonormal = project_orthonormal(
        vec_orig_space, subspace, result_in_orig_space=False
    )
    assert vec_backprojected == pytest.approx(vec_subspace, abs=1e-6)
    assert vec_backprojected_orthonormal == pytest.approx(vec_subspace, abs=1e-6)


def test_compare_project_project_orthonormal():
    random_vectors = torch.randn(100, 50)
    subspace, _ = torch.linalg.qr(random_vectors)
    vec = torch.randn(100, 1)
    assert project_orthonormal(
        vec, subspace, result_in_orig_space=False
    ) == pytest.approx(project(vec, subspace, result_in_orig_space=False), abs=1e-5)
    assert project_orthonormal(
        vec, subspace, result_in_orig_space=True
    ) == pytest.approx(project(vec, subspace, result_in_orig_space=True), abs=1e-5)


def test_flatten_unflatten():
    agent = stable_baselines3.ppo.PPO(
        "MlpPolicy", gym.make("Pendulum-v1"), device="cpu"
    )
    params_flattened = flatten_parameters(agent.policy.parameters())
    params_unflattened = unflatten_parameters_for_agent(params_flattened, agent)
    for p_unfl, p_orig in zip(params_unflattened, agent.policy.parameters()):
        assert torch.all(p_unfl == p_orig)


def test_ppo_gradient():
    env = gym.make("Pendulum-v1")
    agent = stable_baselines3.ppo.PPO(
        "MlpPolicy", DummyVecEnv([lambda: env]), device="cpu"
    )
    rollout_buffer = RolloutBuffer(
        5000, env.observation_space, env.action_space, device="cpu"
    )
    fill_rollout_buffer(env, agent, rollout_buffer)
    combined_gradient_not_full, _, _ = actor_critic_gradient(
        agent, next(rollout_buffer.get())
    )
    combined_gradient_full, _, _ = actor_critic_gradient(
        agent, next(rollout_buffer.get()), all_gradients_fullsize=True
    )
    for g_nf, g_f in zip(combined_gradient_not_full, combined_gradient_full):
        assert g_nf == pytest.approx(g_f, rel=1e-5, abs=1e-5)


def test_combine_actor_critic_parameter_vectors():
    env = gym.make("Pendulum-v1")
    for agent in [
        stable_baselines3.PPO("MlpPolicy", env, device="cpu"),
        stable_baselines3.SAC("MlpPolicy", env, device="cpu"),
    ]:
        policy_params, vf_params = get_actor_critic_parameters(agent)
        params_combined = combine_actor_critic_parameter_vectors(
            flatten_parameters(policy_params), flatten_parameters(vf_params), agent
        )
        assert (
            params_combined == flatten_parameters(get_trained_parameters(agent))
        ).all()


def test_replay_buffer_checkpointing():
    env = gym.make("Pendulum-v1")
    algo = stable_baselines3.SAC(
        "MlpPolicy", env, buffer_size=20, policy_kwargs={"net_arch": [32, 32]}
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
            algo = stable_baselines3.SAC.load(checkpoint)
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
