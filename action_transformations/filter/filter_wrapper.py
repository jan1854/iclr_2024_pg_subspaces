import gym
import numpy as np

from action_transformations.action_transformation_wrapper import ActionTransformationWrapper
from action_transformations.filter.lowpass import Lowpass


class FilterWrapper(ActionTransformationWrapper):
    def __init__(self, env: gym.Env, filt: Lowpass):
        super().__init__(env)
        assert isinstance(env.observation_space, gym.spaces.Box)
        assert isinstance(env.action_space, gym.spaces.Box)
        obs_low = np.concatenate((env.observation_space.low, env.action_space.low))
        obs_high = np.concatenate((env.observation_space.high, env.action_space.high))
        self._observation_space = gym.spaces.Box(obs_low, obs_high)
        self.filter = filt

    def reset_transformation(self, **kwargs) -> None:
        self.filter.reset()

    def transformation_observation(self) -> np.ndarray:
        return self.filter.state

    def transform_action(self, action: np.ndarray) -> np.ndarray:
        return self.filter.filter(action)

    def get_transformation_parameters(self) -> np.ndarray:
        return self.filter.state

    def set_transformation_parameters(self, parameters: np.ndarray) -> None:
        self.filter.state = parameters
