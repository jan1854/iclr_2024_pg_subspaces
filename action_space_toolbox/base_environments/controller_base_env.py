import abc

import gym
import numpy as np


class ControllerBaseEnv(gym.Env, abc.ABC):
    """
    This class is an interface that provides a unified way to extract information on the state of the controllable
    dofs (e.g. position and velocity). This is needed to implement different control modes as generic wrappers that
    work for different environments. Each supported environment needs an implementation of this interface.
    """

    @property
    @abc.abstractmethod
    def dof_positions(self) -> np.ndarray:
        """
        The position of each controllable dof.
        """
        pass

    @property
    @abc.abstractmethod
    def dof_velocities(self) -> np.ndarray:
        """
        The velocity of each controllable dof.
        """
        pass

    @property
    @abc.abstractmethod
    def dof_pos_bounds(self) -> np.ndarray:
        """
        The position bounds of each controllable dof (as a [[low_0, high_0], [low_1, high_1], ...] array).
        """
        pass

    @property
    @abc.abstractmethod
    def dof_vel_bounds(self) -> np.ndarray:
        """
        The velocity bounds of each controllable dof (as a [[low_0, high_0], [low_1, high_1], ...] array).
        """
        pass

    @property
    @abc.abstractmethod
    def dofs_revolute(self) -> np.ndarray:
        """
        Whether the dofs are revolute joints.
        """
        pass
