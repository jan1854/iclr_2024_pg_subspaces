from typing import Union

import gym
import numpy as np

from action_space_toolbox.action_transformation_wrapper import (
    ActionTransformationWrapper,
)
from action_space_toolbox.control_modes.check_wrapped_dof_information import (
    check_wrapped_dof_information,
)


class VelocityControlWrapper(ActionTransformationWrapper):
    def __init__(self, env: gym.Env, gains: Union[float, np.ndarray] = 1.0):
        assert check_wrapped_dof_information(env)
        super().__init__(env)
        if np.isscalar(gains):
            gains = gains * np.ones(env.action_space.shape)
        assert gains.shape == env.action_space.shape
        self.gains = gains
        self.action_space = gym.spaces.Box(
            env.dof_vel_bounds[:, 0].astype(np.float32),  # type: ignore
            env.dof_vel_bounds[:, 1].astype(np.float32),  # type: ignore
        )

    def transform_action(self, action: np.ndarray) -> np.ndarray:
        vel = self.dof_velocities
        return -self.gains * (vel - action)
