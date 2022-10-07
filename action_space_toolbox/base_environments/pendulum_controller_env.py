import numpy as np
from gym.envs.classic_control import PendulumEnv

from action_space_toolbox.base_environments.controller_base_env import ControllerBaseEnv


class PendulumControllerEnv(PendulumEnv, ControllerBaseEnv):
    @property
    def dof_positions(self) -> np.ndarray:
        return np.array([self.state[0]])

    @property
    def dof_velocities(self) -> np.ndarray:
        return np.array([self.state[1]])

    @property
    def dof_pos_bounds(self) -> np.ndarray:
        return np.array([[-np.pi, np.pi]])

    @property
    def dof_vel_bounds(self) -> np.ndarray:
        return np.array([[-self.max_speed, self.max_speed]])

    @property
    def dofs_revolute(self) -> np.ndarray:
        return np.array([True])
