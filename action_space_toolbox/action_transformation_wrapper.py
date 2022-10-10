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
        action_transformed = self.transform_action(action)
        action_transformed = np.clip(action_transformed, self.env.action_space.low, self.env.action_space.high)
        obs, reward, done, info = self.env.step(action_transformed)
        return self.transform_state(obs), reward, done, info

    def reset_transformation(self) -> None:
        return

    def transformation_observation(self) -> np.ndarray:
        return np.array([])

    def transform_state(self, state: np.ndarray) -> np.ndarray:
        return np.concatenate((state, self.transformation_observation()))

    @abc.abstractmethod
    def transform_action(self, action: np.ndarray) -> np.ndarray:
        pass
