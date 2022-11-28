import logging
from typing import Optional, Tuple

import gym
import numpy as np

from action_space_toolbox.control_modes.position_control.abstract_position_control_wrapper import (
    AbstractPositionControlWrapper,
)

logger = logging.getLogger(__name__)


class VariableGainsPositionControlWrapper(AbstractPositionControlWrapper):
    # TODO: In (Bogdanovic, 2020: Learning Variable Impedance Control for Contact Sensitive Tasks) they use an
    #  additional reward term ("trajectory tracking reward") when applying this actuation method

    def __init__(
        self,
        env: gym.Env,
        p_gains_limits: Tuple[float, float],
        d_gains_limits: Optional[Tuple[float, float]] = None,
        positions_relative: bool = False,
        target_position_limits: Optional[np.ndarray] = None,
        controller_steps: int = 1,
        keep_base_timestep: bool = True,
    ):
        super().__init__(
            env,
            positions_relative,
            target_position_limits,
            controller_steps,
            keep_base_timestep,
        )

        self.use_d_gain_sqrt_heuristic = d_gains_limits is None

        assert len(env.action_space.shape) == 1
        action_limits = np.concatenate(
            (
                self.target_position_limits,
                np.array([p_gains_limits] * env.action_space.shape[0]),
            ),
        )
        if not self.use_d_gain_sqrt_heuristic:
            action_limits = np.concatenate(
                (
                    action_limits,
                    np.array([d_gains_limits] * env.action_space.shape[0]),
                ),
            )

        self.action_space = gym.spaces.Box(
            action_limits[:, 0].astype(np.float32),
            action_limits[:, 1].astype(np.float32),
        )

    def _get_target(self, action: np.ndarray) -> np.ndarray:
        return action[: self.env.action_space.shape[0]]

    def _get_controller_gains(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        action_size = self.env.action_space.shape[0]
        p_gains = action[action_size : 2 * action_size]
        if self.use_d_gain_sqrt_heuristic:
            d_gains = np.sqrt(p_gains)
        else:
            d_gains = action[2 * action_size : 3 * action_size]
        return p_gains, d_gains
