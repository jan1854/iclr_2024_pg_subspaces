import logging
from typing import Sequence, Union, Optional

import gym
import numpy as np

from action_space_toolbox.control_modes.velocity_control.abstract_velocity_control_wrapper import (
    AbstractVelocityControlWrapper,
)

logger = logging.getLogger(__name__)


class FixedGainsVelocityControlWrapper(AbstractVelocityControlWrapper):
    def __init__(
        self,
        env: gym.Env,
        gains: Union[float, Sequence[float]],
        target_velocity_limits: Optional[np.ndarray] = None,
        controller_steps: int = 1,
        keep_base_timestep: bool = True,
    ):
        super().__init__(
            env,
            target_velocity_limits,
            controller_steps,
            keep_base_timestep,
        )

        if np.isscalar(gains):
            gains = gains * np.ones(env.action_space.shape)
        gains = np.asarray(gains)
        assert gains.shape == env.action_space.shape
        self.gains = gains
        logger.info(f"Using gains: {gains}.")

        self.action_space = gym.spaces.Box(
            self.target_velocity_limits[:, 0].astype(np.float32),
            self.target_velocity_limits[:, 1].astype(np.float32),
        )

    def _get_target(self, action: np.ndarray) -> np.ndarray:
        return action

    def _get_controller_gains(self, action: np.ndarray) -> np.ndarray:
        return self.gains
