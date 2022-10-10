from typing import Union

import gym
import numpy as np

from action_space_toolbox.action_transformation_wrapper import ActionTransformationWrapper
from action_space_toolbox.base_environments.controller_base_env import ControllerBaseEnv
from action_space_toolbox.util.angles import normalize_angle


class PositionControlWrapper(ActionTransformationWrapper):
    def __init__(self, env: ControllerBaseEnv, p_gains: Union[float, np.ndarray] = 1.0,
                 d_gains: Union[float, np.ndarray] = 1.0):
        super().__init__(env)
        if np.isscalar(p_gains):
            p_gains = p_gains * np.ones(env.action_space.shape)
        if np.isscalar(d_gains):
            d_gains = d_gains * np.ones(env.action_space.shape)
        assert p_gains.shape == d_gains.shape == env.action_space.shape
        self.p_gains = p_gains
        self.d_gains = d_gains
        self.action_space = gym.spaces.Box(env.dof_pos_bounds[:, 0], env.dof_pos_bounds[:, 1])

    def transform_action(self, action: np.ndarray) -> np.ndarray:
        pos = self.dof_positions
        vel = self.dof_velocities
        # Normalize the angle distances of all revolute dofs (so that diff(pi - 0.1, -pi + 0.1) = 0.2 not 2pi - 0.2)
        pos_error = ~self.dofs_revolute * (pos - action) + self.dofs_revolute * normalize_angle(pos - action)
        return -self.p_gains * pos_error - self.d_gains * vel
