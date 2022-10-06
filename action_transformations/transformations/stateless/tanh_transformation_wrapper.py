from typing import Sequence, Union

import gym
import numpy as np

from action_transformations.action_transformation_wrapper import ActionTransformationWrapper


class TanhTransformationWrapper(ActionTransformationWrapper):
    def __init__(self, env: gym.Env, scaling: Union[float, Sequence[float]]):
        super().__init__(env)
        self.scaling_pre = np.array(scaling) * np.ones(env.action_space.shape)    # Copy to array if scalar
        assert isinstance(env.action_space, gym.spaces.Box)
        self.offset = 0.5 * (env.action_space.low + env.action_space.high)
        tanh_output_range = np.tanh(env.action_space.high * self.scaling_pre) \
                            - np.tanh(env.action_space.low * self.scaling_pre)
        self.scaling_post = (self.action_space.high - self.action_space.low) / tanh_output_range

    def get_transformation_parameters(self) -> np.ndarray:
        return self.scaling_pre

    def set_transformation_parameters(self, transformation_parameters: np.ndarray) -> None:
        self.scaling_pre = transformation_parameters

    def transform_action(self, action: np.ndarray) -> np.ndarray:
        return np.tanh((action - self.offset) * self.scaling_pre) * self.scaling_post + self.offset
