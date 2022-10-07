import abc

import gym
import numpy as np


class ControllerBaseEnv(gym.Env, abc.ABC):
    @property
    @abc.abstractmethod
    def dof_positions(self) -> np.ndarray:
        pass

    @property
    @abc.abstractmethod
    def dof_velocities(self) -> np.ndarray:
        pass

    @property
    @abc.abstractmethod
    def dof_pos_bounds(self) -> np.ndarray:
        pass

    @property
    @abc.abstractmethod
    def dof_vel_bounds(self) -> np.ndarray:
        pass

    @property
    @abc.abstractmethod
    def dofs_revolute(self) -> np.ndarray:
        pass
