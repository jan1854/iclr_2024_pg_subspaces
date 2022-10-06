import gym
import numpy as np

from action_space_toolbox.action_transformation_wrapper import ActionTransformationWrapper
from action_space_toolbox.control_modes.get_control_state import get_control_state


class VelocityControlWrapper(ActionTransformationWrapper):
    def __init__(self, env: gym.Env, gains: np.ndarray):
        super().__init__(env)
        assert gains.shape == gains.shape == env.action_space.shape
        self.gains = gains

    def transform_action(self, action: np.ndarray) -> np.ndarray:
        _, vel = get_control_state(self.env)
        return -self.gains * (vel - action)
        # return np.zeros(1)
