import logging
from typing import Union, Optional, Sequence

import gym
import numpy as np

from action_space_toolbox.controller_base.controller_base_wrapper import (
    ControllerBaseWrapper,
)
from action_space_toolbox.action_transformation_wrapper import (
    ActionTransformationWrapper,
)
from action_space_toolbox.control_modes.check_wrapped import (
    check_wrapped,
)

logger = logging.getLogger(__name__)


class VelocityControlWrapper(ActionTransformationWrapper):
    def __init__(
        self,
        env: gym.Env,
        gains: Union[float, Sequence[float]] = 1.0,
        target_velocity_limits: Optional[
            Union[Sequence[float], Sequence[Sequence[float]]]
        ] = None,
        controller_steps: int = 1,
        keep_base_timestep: bool = True,
    ):
        assert check_wrapped(env, ControllerBaseWrapper)
        super().__init__(env, controller_steps, keep_base_timestep)
        if np.isscalar(gains):
            gains = gains * np.ones(env.action_space.shape)
        gains = np.asarray(gains)
        assert gains.shape == env.action_space.shape
        self.gains = gains
        logger.info(f"Using gains: {gains}.")
        if target_velocity_limits is not None:
            target_velocity_limits = np.asarray(target_velocity_limits)
            # Assume that the limits are the same for each action dimension if a single (low, high) pair is passed
        elif env.actuator_vel_bounds is not None:  # type: ignore
            target_velocity_limits = env.actuator_vel_bounds  # type: ignore
        else:
            target_velocity_limits = np.array([-10.0, 10.0])
            logger.info(
                f"Did not find target velocity limits for environment {env}. Using the default [-10, 10]."
            )

        if target_velocity_limits.ndim == 1:
            target_velocity_limits = target_velocity_limits[None].repeat(
                env.action_space.shape, axis=0
            )
        self.action_space = gym.spaces.Box(
            target_velocity_limits[:, 0].astype(np.float32),
            target_velocity_limits[:, 1].astype(np.float32),
        )

    def transform_action(self, action: np.ndarray) -> np.ndarray:
        vel = self.actuator_velocities
        return (-self.gains * (vel - action)).astype(np.float32)
