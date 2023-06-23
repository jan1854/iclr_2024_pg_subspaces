import logging
from typing import Optional, Dict, Tuple

import gym
import numpy as np

from action_space_toolbox.controller_base.controller_base_wrapper import (
    ControllerBaseWrapper,
)
from sb3_utils.common.training import check_wrapped

logger = logging.getLogger(__name__)


class OptimalPositionControlWrapper(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        positions_relative: bool = False,
        target_position_limits: Optional[np.ndarray] = None,
    ):
        assert check_wrapped(env, ControllerBaseWrapper)
        super().__init__(env)
        self.positions_relative = positions_relative

        if target_position_limits is not None:
            target_position_limits = np.asarray(target_position_limits)
            # Assume that the limits are the same for each action dimension if a single (low, high) pair is passed
            if target_position_limits.ndim == 1:
                target_position_limits[None].repeat(env.action_space.shape, axis=0)
        else:
            if positions_relative:
                # TODO: This assumes revolute joints with no joint limits
                assert np.all(self.actuators_revolute)
                target_position_limits = np.stack(
                    (
                        -np.pi * np.ones(env.action_space.shape, dtype=np.float32),
                        np.pi * np.ones(env.action_space.shape, dtype=np.float32),
                    ),
                    axis=1,
                )
            else:
                target_position_limits = env.actuator_pos_bounds  # type: ignore

        self.action_space = gym.spaces.Box(
            target_position_limits[:, 0].astype(np.float32),
            target_position_limits[:, 1].astype(np.float32),
        )
        self.env.set_timestep(0.0)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        # TODO: Relative positions not supported yet
        self.env.set_actuator_states(action, np.zeros_like(action))
        return self.env.step(np.zeros_like(action))
