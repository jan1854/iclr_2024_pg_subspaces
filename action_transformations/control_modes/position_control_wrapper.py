import gym
import numpy as np

from action_transformations.action_transformation_wrapper import ActionTransformationWrapper
from action_transformations.control_modes.get_control_state import get_control_state
from action_transformations.util.angles import normalize_angle


class PositionControlWrapper(ActionTransformationWrapper):
    def __init__(self, env: gym.Env, p_gains: np.ndarray, d_gains: np.ndarray, pos_is_angle: bool):
        super().__init__(env)
        assert p_gains.shape == d_gains.shape == env.action_space.shape
        self.p_gains = p_gains
        self.d_gains = d_gains
        self.pos_is_angle = pos_is_angle

    def transform_action(self, action: np.ndarray) -> np.ndarray:
        pos, vel = get_control_state(self.env)
        pos_error = pos - action if not self.pos_is_angle else normalize_angle(pos - action)
        return -self.p_gains * pos_error - self.d_gains * vel
