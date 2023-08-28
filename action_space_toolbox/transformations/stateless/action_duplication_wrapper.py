from typing import Sequence, Union, Optional

import gym
import numpy as np

from action_space_toolbox.action_transformation_wrapper import (
    ActionTransformationWrapper,
)


class ActionDuplicationWrapper(ActionTransformationWrapper):
    def __init__(self, env: gym.Env, num_action_duplicates: int, rescale_weights: bool = True,
                 weights: Optional[Sequence[float]] = None):
        super().__init__(env)
        original_action_space = env.action_space
        assert len(original_action_space.shape) == 1
        assert isinstance(original_action_space, gym.spaces.Box)
        self.action_space = gym.spaces.Box(np.repeat(original_action_space.low, num_action_duplicates),
                                           np.repeat(original_action_space.high, num_action_duplicates),
                                           dtype=original_action_space.dtype)
        self.num_action_duplicates = num_action_duplicates
        self.weights = np.asarray(weights) if weights is not None else np.array([1] * num_action_duplicates)
        if rescale_weights:
            self.weights = self.weights / np.sum(self.weights)
        self.weights = np.tile(self.weights, original_action_space.shape[0])

    def transform_action(self, action: np.ndarray) -> np.ndarray:
        action_weighted = self.weights * action
        action_original_space = action_weighted.reshape(-1, self.num_action_duplicates).sum(axis=1)
        return np.clip(action_original_space, self.env.action_space.low, self.env.action_space.high)
