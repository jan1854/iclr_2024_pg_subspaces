import gym
import numpy as np

from action_space_toolbox.control_modes.get_control_state import get_control_state
from action_space_toolbox.control_modes.position_control_wrapper import PositionControlWrapper
from action_space_toolbox.control_modes.velocity_control_wrapper import VelocityControlWrapper
from action_space_toolbox.util.angles import normalize_angle


def test_position_control_pendulum():
    env = gym.make("Pendulum-v1")
    env = PositionControlWrapper(env, np.array([15.0]), np.array([2.0]), True)
    env.reset()
    target_positions = [np.pi - d for d in np.arange(0.0, 0.5, 0.1)] + [-np.pi + d for d in np.arange(0.1, 0.5, 0.1)]
    for target_position in target_positions:
        for _ in range(100):
            env.step(target_position)
        for _ in range(10):
            env.step(target_position)
            assert np.all(normalize_angle(get_control_state(env)[0] - target_position) < 0.1)


def test_velocity_control_pendulum():
    env = gym.make("Pendulum-v1", g=0.0)    # Disable gravity, so that we can achieve the velocity at all times
    env = VelocityControlWrapper(env, np.array([10.0]))
    env.reset()
    target_velocities = np.arange(-0.2, 0.2, 0.1)
    for target_velocity in target_velocities:
        for _ in range(100):
            env.step(target_velocity)
            env.render()
        for _ in range(10):
            env.step(target_velocity)
            assert np.all(np.abs(get_control_state(env)[1] - target_velocity).item() < 0.1)
