from typing import Union

import numpy as np

from action_space_toolbox.action_transformation_wrapper import ActionTransformationWrapper
from action_space_toolbox.base_environments.controller_base_env import ControllerBaseEnv


class VelocityControlWrapper(ActionTransformationWrapper):
    def __init__(self, env: ControllerBaseEnv, gains: Union[float, np.ndarray] = 1.0):
        super().__init__(env)
        if np.isscalar(gains):
            gains = gains * np.ones(env.action_space.shape)
        assert gains.shape == env.action_space.shape
        self.gains = gains

    def transform_action(self, action: np.ndarray) -> np.ndarray:
        vel = self.dof_velocities
        return -self.gains * (vel - action)
