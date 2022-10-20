import gym
import numpy as np
from _pytest.python_api import approx

from action_space_toolbox import PositionControlWrapper, DofInformationWrapper
from action_space_toolbox.util.angles import normalize_angle


def test_position_control_pendulum():
    env = gym.make("Pendulum_PC-v1")
    env.reset()
    target_positions = [np.pi - d for d in np.arange(0.0, 0.5, 0.1)] + [
        -np.pi + d for d in np.arange(0.1, 0.5, 0.1)
    ]
    for target_position in target_positions:
        for _ in range(100):
            env.step(target_position)
        for _ in range(10):
            env.step(target_position)
            assert np.all(normalize_angle(env.dof_positions[0] - target_position) < 0.1)


def test_velocity_control_pendulum():
    env = gym.make(
        "Pendulum_VC-v1",
        g=0.0,  # Disable gravity, so that we can achieve the velocity at all times
    )
    env.reset()
    target_velocities = np.arange(-0.2, 0.2, 0.1)
    for target_velocity in target_velocities:
        for _ in range(100):
            env.step(target_velocity)
        for _ in range(10):
            env.step(target_velocity)
            assert np.all(np.abs(env.dof_velocities[0] - target_velocity).item() < 0.1)


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

    env = PositionControlWrapper(DummyDofInformationWrapper(DummyEnv()), p_gains=1.0, d_gains=0.0)  # type: ignore
    pos_diff = env.transform_action(np.array([-np.pi + 0.1, -2.9]))
    assert pos_diff == approx(np.array([0.2, -5.8]))
