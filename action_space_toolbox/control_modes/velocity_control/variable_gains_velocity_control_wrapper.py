import logging
from typing import Optional, Tuple

import gym
import numpy as np

from action_space_toolbox.control_modes.velocity_control.abstract_velocity_control_wrapper import (
    AbstractVelocityControlWrapper,
)

logger = logging.getLogger(__name__)


class VariableGainsVelocityControlWrapper(AbstractVelocityControlWrapper):
    # TODO: In (Bogdanovic, 2020: Learning Variable Impedance Control for Contact Sensitive Tasks) they use an
    #  additional reward term ("trajectory tracking reward") for their variable gain position controller

    def __init__(
        self,
        env: gym.Env,
        gains_limits: Tuple[float, float],
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

        assert len(env.action_space.shape) == 1
        action_limits = np.concatenate(
            (
                self.target_velocity_limits,
                np.array([gains_limits] * env.action_space.shape[0]),
            ),
        )

        self.action_space = gym.spaces.Box(
            action_limits[:, 0].astype(np.float32),
            action_limits[:, 1].astype(np.float32),
        )

    def _get_target(self, action: np.ndarray) -> np.ndarray:
        return action[: self.env.action_space.shape[0]]

    def _get_controller_gains(self, action: np.ndarray) -> np.ndarray:
        gains = action[self.env.action_space.shape[0] :]
        return gains
