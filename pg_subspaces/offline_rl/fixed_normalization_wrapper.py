import gym.core
import numpy as np


class FixedNormalizationWrapper(gym.core.ObservationWrapper):
    def __init__(self, env: gym.Env, mean: np.ndarray, std: np.ndarray, eps: float):
        super().__init__(env)
        self.mean = mean
        self.std = std
        self.eps = eps

    def observation(self, observation: np.ndarray) -> np.ndarray:
        return (observation - self.mean) / (self.std + self.eps)
