import abc
from typing import Optional

import gym
import numpy as np


class ControllerBaseWrapper(gym.Wrapper):
    """
    This wrapper is an interface that provides a unified way to extract information on the state of the controllable
    actuator (e.g. position and velocity). This is needed to implement different control modes as generic wrappers that
    work for different environments. Each supported environment needs an implementation of this interface.
    """

    def __init__(
        self,
        env: gym.Env,
        actuator_pos_bounds: Optional[np.ndarray],
        actuator_vel_bounds: Optional[np.ndarray],
        actuators_revolute: np.ndarray,
    ):
        super().__init__(env)
        self.actuator_pos_bounds = actuator_pos_bounds
        self.actuator_vel_bounds = actuator_vel_bounds
        self.actuators_revolute = actuators_revolute

    @property
    @abc.abstractmethod
    def actuator_positions(self) -> np.ndarray:
        """
        The position of each controllable actuator.
        """

    @property
    @abc.abstractmethod
    def actuator_velocities(self) -> np.ndarray:
        """
        The velocity of each controllable actuator.
        """

    @abc.abstractmethod
    def set_actuator_states(
        self, positions: np.ndarray, velocities: np.ndarray
    ) -> None:
        pass

    @property
    @abc.abstractmethod
    def timestep(self) -> float:
        """
        The time step of the simulation.
        """

    @abc.abstractmethod
    def set_timestep(self, timestep: float) -> None:
        """
        Set the time step of the simulation.
        """

    @property
    def base_env_timestep_factor(self) -> int:
        return 1
