import abc
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
from action_space_toolbox.util.angles import normalize_angle


class AbstractPositionControlWrapper(ActionTransformationWrapper, abc.ABC):
    def __init__(
        self,
        env: gym.Env,
        positions_relative: bool = False,
        target_position_limits: Optional[np.ndarray] = None,
        controller_steps: int = 1,
        keep_base_timestep: bool = True,
    ):
        assert check_wrapped(env, ControllerBaseWrapper)
        super().__init__(env, controller_steps, keep_base_timestep)

        self.positions_relative = positions_relative

        if target_position_limits is not None:
            self.target_position_limits = np.asarray(target_position_limits)
            # Assume that the limits are the same for each action dimension if a single (low, high) pair is passed
            if target_position_limits.ndim == 1:
                self.target_position_limits[None].repeat(env.action_space.shape, axis=0)
        else:
            if positions_relative:
                # TODO: This assumes revolute joints with no joint limits
                assert np.all(self.actuators_revolute)
                self.target_position_limits = np.stack(
                    (
                        -np.pi * np.ones(env.action_space.shape, dtype=np.float32),
                        np.pi * np.ones(env.action_space.shape, dtype=np.float32),
                    ),
                    axis=1,
                )
            else:
                self.target_position_limits = env.actuator_pos_bounds  # type: ignore

    def transform_action(self, action: np.ndarray) -> np.ndarray:
        pos_error = self._get_target_pos_error(action)
        p_gains, d_gains = self._get_controller_gains(action)
        return (p_gains * pos_error - d_gains * self.actuator_velocities).astype(
            self.env.action_space.dtype
        )

    def _get_target_pos_error(self, action: np.ndarray) -> np.ndarray:
        target = self._get_target(action)
        if self.positions_relative:
            pos_error = target
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
                normalize_angle(target - self.actuator_positions),
                target - self.actuator_positions,
            )
        return pos_error

    @abc.abstractmethod
    def _get_target(self, action: np.ndarray) -> np.ndarray:
        pass

    @abc.abstractmethod
    def _get_controller_gains(self, action: np.ndarray) -> np.ndarray:
        pass
