from typing import Union, Optional, Sequence

import gym
import numpy as np

from action_space_toolbox import DofInformationWrapper
from action_space_toolbox.action_transformation_wrapper import (
    ActionTransformationWrapper,
)
from action_space_toolbox.control_modes.check_wrapped import (
    check_wrapped,
)


class VelocityControlWrapper(ActionTransformationWrapper):
    def __init__(
        self,
        env: gym.Env,
        gains: Union[float, Sequence[float]] = 1.0,
        target_velocity_limits: Optional[
            Union[Sequence[float], Sequence[Sequence[float]]]
        ] = None,
    ):
        assert check_wrapped(env, DofInformationWrapper)
        super().__init__(env)
        if np.isscalar(gains):
            gains = gains * np.ones(env.action_space.shape)
        gains = np.asarray(gains)
        assert gains.shape == env.action_space.shape
        self.gains = gains
        if target_velocity_limits is not None:
            target_velocity_limits = np.asarray(target_velocity_limits)
            # Assume that the limits are the same for each action dimension if a single (low, high) pair is passed
            if target_velocity_limits.ndim == 1:
                target_velocity_limits = target_velocity_limits[None].repeat(
                    env.action_space.shape, axis=0
                )
        else:
            target_velocity_limits = env.dof_vel_bounds  # type: ignore
        self.action_space = gym.spaces.Box(
            target_velocity_limits[:, 0].astype(np.float32),
            target_velocity_limits[:, 1].astype(np.float32),
        )

    def transform_action(self, action: np.ndarray) -> np.ndarray:
        vel = self.dof_velocities
        return -self.gains * (vel - action)
