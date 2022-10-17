import abc
from typing import Optional

import gym
import numpy as np


class DofInformationWrapper(gym.Wrapper):
    """
    This wrapper is an interface that provides a unified way to extract information on the state of the controllable
    dofs (e.g. position and velocity). This is needed to implement different control modes as generic wrappers that
    work for different environments. Each supported environment needs an implementation of this interface.
    """

    def __init__(
        self,
        env: gym.Env,
        dof_pos_bounds: Optional[np.ndarray],
        dof_vel_bounds: Optional[np.ndarray],
        dofs_revolute: np.ndarray,
    ):
        super().__init__(env)
        self.dof_pos_bounds = dof_pos_bounds
        self.dof_vel_bounds = dof_vel_bounds
        self.dofs_revolute = dofs_revolute

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
