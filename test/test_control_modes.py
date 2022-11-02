import gym
import numpy as np
from _pytest.python_api import approx
from dmc2gym.wrappers import DMCWrapper

import action_space_toolbox
from action_space_toolbox import PositionControlWrapper, DofInformationWrapper
from action_space_toolbox.util.angles import normalize_angle


def execute_episode(env: gym.Env) -> int:
    env.reset()
    done = False
    i = 0
    while not done:
        i += 1
        _, _, done, _ = env.step(env.action_space.sample())
    return i


def test_position_control_pendulum():
    env = gym.make("Pendulum_PC-v1", normalize=False)
    target_positions = [np.pi - d for d in np.arange(0.0, 0.2, 0.02)] + [
        -np.pi + d for d in np.arange(0.02, 0.2, 0.02)
    ]
    for target_position in target_positions:
        env.reset()
        for _ in range(100):
            env.step(target_position)
        for _ in range(10):
            env.step(target_position)
            assert np.all(
                normalize_angle(env.dof_positions[0] - target_position) < 0.05
            )


def test_velocity_control_pendulum():
    env = gym.make(
        "Pendulum_VC-v1",
        g=0.0,  # Disable gravity, so that we can achieve the velocity at all times
        normalize=False,
    )
    target_velocities = np.arange(-0.2, 0.2, 0.05)
    for target_velocity in target_velocities:
        env.reset()
        for _ in range(100):
            env.step(target_velocity)
        for _ in range(10):
            env.step(target_velocity)
            assert np.all(np.abs(env.dof_velocities[0] - target_velocity).item() < 0.05)


def test_position_control_multiturn():
    class DummyEnv:
        action_space = gym.spaces.Box(
            np.array([-10, -10], dtype=np.float32),
            np.array([10, 10], dtype=np.float32),
        )

    class DummyDofInformationWrapper(DofInformationWrapper):
        def __init__(self, env: gym.Env):
            dofs_revolute = np.array([True, True])
            super().__init__(
                env,
                np.array([[-np.pi, np.pi], [-3, 3]]),
                np.array([[-1, -1], [1, 1]]),
                dofs_revolute,
            )

        @property
        def dof_positions(self) -> np.ndarray:
            return np.array([np.pi - 0.1, 2.9])

        @property
        def dof_velocities(self) -> np.ndarray:
            return np.zeros(2)

    env = PositionControlWrapper(
        DummyDofInformationWrapper(DummyEnv()),  # type: ignore
        p_gains=1.0,
        d_gains=0.0,
        keep_base_timestep=True,
    )
    pos_diff = env.transform_action(np.array([-np.pi + 0.1, -2.9]))
    assert pos_diff == approx(np.array([0.2, -5.8]))


def test_position_control_dmc_pendulum():
    env = gym.make("dmc_Pendulum-swingup_PC-v1", normalize=False)
    target_positions = [np.pi - d for d in np.arange(0.0, 0.2, 0.02)] + [
        -np.pi + d for d in np.arange(0.02, 0.2, 0.02)
    ]
    for target_position in target_positions:
        env.reset()
        for _ in range(200):
            env.step(target_position)
        for _ in range(20):
            env.step(target_position)
            assert np.all(
                normalize_angle(env.dof_positions[0] - target_position) < 0.05
            ), f"Did not reach target position {target_position}"


def test_velocity_control_dmc_pendulum():
    env = gym.make("dmc_Pendulum-swingup_VC-v1", normalize=False)
    # Disable gravity, so that we can achieve the velocity at all times
    env.unwrapped._env.physics.model.opt.gravity = np.zeros(3)
    target_velocities = np.arange(-0.2, 0.2, 0.05)
    for target_velocity in target_velocities:
        env.reset()
        for _ in range(200):
            env.step(target_velocity)
        for _ in range(20):
            env.step(target_velocity)
            assert np.all(np.abs(env.dof_velocities[0] - target_velocity).item() < 0.05)


# TODO: Make sure that the dm_control environments do not terminate too early
def test_controller_frequency():
    controller_steps = 4
    episode_length = 1000
    for env_id in ["HalfCheetah_PC-v3", "dmc_Acrobot-swingup_PC-v1"]:
        env_original_timestep = gym.make(env_id, controller_steps=1)
        env = gym.make(env_id, controller_steps=controller_steps)
        env.reset()
        steps = execute_episode(env)
        if isinstance(env.unwrapped, DMCWrapper):
            sim_time = env.physics.data.time
            frame_skip = 1
        else:
            sim_time = env.sim.data.time
            frame_skip = env.frame_skip
        assert env.timestep == approx(env_original_timestep.timestep / controller_steps)
        assert steps == episode_length
        assert sim_time == approx(env.timestep * steps * frame_skip * controller_steps)
