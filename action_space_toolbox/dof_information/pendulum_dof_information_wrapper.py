import gym.envs.classic_control
import numpy as np

from action_space_toolbox.dof_information.dof_information_wrapper import (
    DofInformationWrapper,
)
from action_space_toolbox.util.angles import normalize_angle


class PendulumDofInformationWrapper(DofInformationWrapper):
    def __init__(self, env: gym.Env):
        assert isinstance(env.unwrapped, gym.envs.classic_control.PendulumEnv)
        super().__init__(
            env,
            np.array([[-np.pi, np.pi]]),
            np.array([[-env.max_speed, env.max_speed]]),
            np.array([True]),
        )

    @property
    def dof_positions(self) -> np.ndarray:
        return np.array([normalize_angle(self.state[0])])

    @property
    def dof_velocities(self) -> np.ndarray:
        return np.array([self.env.state[1]])

    @property
    def timestep(self) -> float:
        return self.env.dt

    def set_timestep(self, timestep: float) -> None:
        self.env.dt = timestep
