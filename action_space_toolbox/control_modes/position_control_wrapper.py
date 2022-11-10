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
from action_space_toolbox.util.angles import normalize_angle

logger = logging.getLogger(__name__)


class PositionControlWrapper(ActionTransformationWrapper):
    def __init__(
        self,
        env: gym.Env,
        p_gains: Union[float, Sequence[float]] = 1.0,
        d_gains: Union[float, Sequence[float]] = 1.0,
        positions_relative: bool = False,
        target_position_limits: Optional[np.ndarray] = None,
        controller_steps: int = 1,
        keep_base_timestep: bool = True,
    ):
        assert check_wrapped(env, ControllerBaseWrapper)
        super().__init__(env, controller_steps, keep_base_timestep)
        if np.isscalar(p_gains):
            p_gains = p_gains * np.ones(env.action_space.shape)
        if np.isscalar(d_gains):
            d_gains = d_gains * np.ones(env.action_space.shape)
        p_gains = np.asarray(p_gains)
        d_gains = np.asarray(d_gains)
        assert p_gains.shape == d_gains.shape == env.action_space.shape
        self.p_gains = p_gains
        self.d_gains = d_gains
        logger.info(f"Using p_gains: {p_gains} and d_gains: {d_gains}.")
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

    def transform_action(self, action: np.ndarray) -> np.ndarray:
        pos_error = self._get_target_pos_error(action)
        return (
            self.p_gains * pos_error - self.d_gains * self.actuator_velocities
        ).astype(self.env.action_space.dtype)

    def _get_target_pos_error(self, action: np.ndarray) -> np.ndarray:
        if self.positions_relative:
            pos_error = action
        else:
            # Normalize the angle distances of multi-turn revolute actuators (so that diff(pi - 0.1, -pi + 0.1) = 0.2
            # not 2pi - 0.2)
            multi_turn = np.logical_and(
                np.logical_and(
                    self.actuators_revolute, self.actuator_pos_bounds[:, 0] == -np.pi
                ),
                self.actuator_pos_bounds[:, 1] == np.pi,
            )
            pos_error = np.where(
                multi_turn,
                normalize_angle(action - self.actuator_positions),
                action - self.actuator_positions,
            )
        return pos_error
