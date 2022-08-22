import gym
import numpy as np

from action_transformations.action_transformation_wrapper import ActionTransformationWrapper


class TanhTransformationWrapper(ActionTransformationWrapper):
    def __init__(self, scaling: np.ndarray, env: gym.Env):
        super().__init__(env)
        self.scaling_pre = scaling
        assert isinstance(env.action_space, gym.spaces.Box)
        self.offset = 0.5 * (env.action_space.low + env.action_space.high)
        self.scaling_post = 0.5 * (self.action_space.high - self.action_space.low)

    def get_transformation_parameters(self) -> np.ndarray:
        return self.scaling_pre

    def set_transformation_parameters(self, transformation_parameters: np.ndarray) -> None:
        self.scaling_pre = transformation_parameters

    def transform_action(self, action: np.ndarray) -> np.ndarray:
        return np.tanh(action * self.scaling_pre) * self.scaling_post + self.offset
