import abc
from typing import Dict, Tuple

import gym
import numpy as np


class ActionTransformationWrapper(gym.Wrapper, abc.ABC):
    """
    An action transformation wrapper similar to gym.core.ActionWrapper. The main difference is that this wrapper allows
    for state-full transformations (requires changing the observations).
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)

    def reset(self, **kwargs) -> np.ndarray:
        self.reset_transformation()
        return self.transform_state(self.env.reset())

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        obs, reward, done, info = self.env.step(self.transform_action(action))
        return self.transform_state(obs), reward, done, info

    def reset_transformation(self) -> None:
        return

    def transformation_observation(self) -> np.ndarray:
        return np.array([])

    def get_transformation_parameters(self) -> np.ndarray:
        return np.array([])

    def set_transformation_parameters(self, transformation_parameters: np.ndarray) -> None:
        return

    def transform_state(self, state: np.ndarray) -> np.ndarray:
        return np.concatenate((state, self.transformation_observation()))

    @abc.abstractmethod
    def transform_action(self, action: np.ndarray) -> np.ndarray:
        pass
