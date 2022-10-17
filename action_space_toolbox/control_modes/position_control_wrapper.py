from typing import Union, Optional

import gym
import numpy as np

from action_space_toolbox.action_transformation_wrapper import (
    ActionTransformationWrapper,
)
from action_space_toolbox.control_modes.check_wrapped_dof_information import (
    check_wrapped_dof_information,
)
from action_space_toolbox.util.angles import normalize_angle


class PositionControlWrapper(ActionTransformationWrapper):
    def __init__(
        self,
        env: gym.Env,
        p_gains: Union[float, np.ndarray] = 1.0,
        d_gains: Union[float, np.ndarray] = 1.0,
        positions_relative: bool = False,
        target_position_limits: Optional[np.ndarray] = None,
    ):
        assert check_wrapped_dof_information(env)
        super().__init__(env)
        if np.isscalar(p_gains):
            p_gains = p_gains * np.ones(env.action_space.shape)
        if np.isscalar(d_gains):
            d_gains = d_gains * np.ones(env.action_space.shape)
        assert p_gains.shape == d_gains.shape == env.action_space.shape
        self.p_gains = p_gains
        self.d_gains = d_gains
        self.positions_relative = positions_relative

        if target_position_limits is not None:
            target_position_limits = np.asarray(target_position_limits)
            # Assume that the limits are the same for each action dimension if a single (low, high) pair is passed
            if target_position_limits.ndim == 1:
                target_position_limits[None].repeat(env.action_space.shape, axis=0)
        else:
            if positions_relative:
                # TODO: This assumes revolute joints with no joint limits
                target_position_limits = np.stack(
                    (
                        -np.pi * np.ones(env.action_space.shape, dtype=np.float32),
                        np.pi * np.ones(env.action_space.shape, dtype=np.float32),
                    ),
                    axis=1,
                )
            else:
                target_position_limits = env.dof_pos_bounds  # type: ignore

        self.action_space = gym.spaces.Box(
            target_position_limits[:, 0].astype(np.float32),
            target_position_limits[:, 1].astype(np.float32),
        )

    def transform_action(self, action: np.ndarray) -> np.ndarray:
        pos = self.dof_positions
        vel = self.dof_velocities
        # Normalize the angle distances of all revolute dofs (so that diff(pi - 0.1, -pi + 0.1) = 0.2 not 2pi - 0.2)
        # TODO: This only works if there are no angle bounds in place and the joint can do an arbitrary number of
        #       rotations
        if self.positions_relative:
            pos_error = action
        else:
            pos_error = ~self.dofs_revolute * (
                pos - action
            ) + self.dofs_revolute * normalize_angle(pos - action)
        return -self.p_gains * pos_error - self.d_gains * vel
