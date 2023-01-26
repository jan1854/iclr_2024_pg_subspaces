import abc
import logging
from typing import Optional

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


class AbstractVelocityControlWrapper(ActionTransformationWrapper, abc.ABC):
    def __init__(
        self,
        env: gym.Env,
        target_velocity_limits: Optional[np.ndarray] = None,
        controller_steps: int = 1,
        keep_base_timestep: bool = True,
    ):
        assert check_wrapped(env, ControllerBaseWrapper)
        super().__init__(env, controller_steps, keep_base_timestep)

        if target_velocity_limits is not None:
            self.target_velocity_limits = target_velocity_limits
        else:
            # Assume that the limits are the same for each action dimension if a single (low, high) pair is passed
            if env.actuator_vel_bounds is not None:  # type: ignore
                self.target_velocity_limits = env.actuator_vel_bounds  # type: ignore
            else:
                self.target_velocity_limits = np.array([-10.0, 10.0])
                logger.debug(
                    f"Did not find target velocity limits for environment {env}. Using the default [-10, 10]."
                )
        # Assume that the limits are the same for each action dimension if a single (low, high) pair is passed
        if self.target_velocity_limits.ndim == 1:
            self.target_velocity_limits = self.target_velocity_limits[None].repeat(
                env.action_space.shape, axis=0
            )

    def transform_action(self, action: np.ndarray) -> np.ndarray:
        vel = self.actuator_velocities
        target = self._get_target(action)
        gains = self._get_controller_gains(action)
        return (gains * (target - vel)).astype(np.float32)

    @abc.abstractmethod
    def _get_target(self, action: np.ndarray) -> np.ndarray:
        pass

    @abc.abstractmethod
    def _get_controller_gains(self, action: np.ndarray) -> np.ndarray:
        pass
