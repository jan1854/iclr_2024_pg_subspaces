import logging
from typing import Sequence, Union, Tuple, Optional

import gym
import numpy as np

from action_space_toolbox.control_modes.position_control.abstract_position_control_wrapper import (
    AbstractPositionControlWrapper,
)

logger = logging.getLogger(__name__)


class FixedGainsPositionControlWrapper(AbstractPositionControlWrapper):
    def __init__(
        self,
        env: gym.Env,
        p_gains: Union[float, Sequence[float]],
        d_gains: Union[float, Sequence[float]],
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

        if np.isscalar(p_gains):
            p_gains = p_gains * np.ones(env.action_space.shape)
        if np.isscalar(d_gains):
            d_gains = d_gains * np.ones(env.action_space.shape)
        p_gains = np.asarray(p_gains)
        d_gains = np.asarray(d_gains)
        assert p_gains.shape == d_gains.shape == env.action_space.shape
        self.p_gains = p_gains
        self.d_gains = d_gains
        logger.debug(f"Using p_gains: {p_gains} and d_gains: {d_gains}.")

        self.action_space = gym.spaces.Box(
            self.target_position_limits[:, 0].astype(np.float32),
            self.target_position_limits[:, 1].astype(np.float32),
        )

    def _get_target(self, action: np.ndarray) -> np.ndarray:
        return action

    def _get_controller_gains(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        return self.p_gains, self.d_gains
