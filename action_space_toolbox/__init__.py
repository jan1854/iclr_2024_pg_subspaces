from typing import Type, TypeVar

import gym.envs.classic_control
import numpy as np
from gym.envs.classic_control import PendulumEnv
from gym.envs.mujoco import MujocoEnv

from action_space_toolbox.dof_information.dof_information_wrapper import (
    DofInformationWrapper,
)
from action_space_toolbox.dof_information.mujoco_dof_information_wrapper import (
    MujocoDofInformationWrapper,
)
from action_space_toolbox.dof_information.pendulum_dof_information_wrapper import (
    PendulumDofInformationWrapper,
)
from action_space_toolbox.control_modes.position_control_wrapper import (
    PositionControlWrapper,
)
from action_space_toolbox.control_modes.velocity_control_wrapper import (
    VelocityControlWrapper,
)

TEnv = TypeVar("TEnv", bound=gym.Env)


def wrap_env_dof_information(env: gym.Env) -> DofInformationWrapper:
    if isinstance(env.unwrapped, PendulumEnv):
        return PendulumDofInformationWrapper(env)
    elif isinstance(env.unwrapped, MujocoEnv):
        return MujocoDofInformationWrapper(env)
    else:
        raise NotImplementedError(
            f"Environment {type(env.unwrapped)} is not supported."
        )


def create_vc_env(
    base_env_type: Type[TEnv], gains: np.ndarray, **kwargs
) -> VelocityControlWrapper:
    return VelocityControlWrapper(
        wrap_env_dof_information(base_env_type(**kwargs)), gains
    )


def create_pc_env(
    base_env_type: Type[TEnv], p_gains: np.ndarray, d_gains: np.ndarray, **kwargs
) -> PositionControlWrapper:
    return PositionControlWrapper(
        wrap_env_dof_information(base_env_type(**kwargs)), p_gains, d_gains
    )


gym.register(
    id="Pendulum_VC-v1",
    entry_point="action_space_toolbox:create_vc_env",
    kwargs={"base_env_type": PendulumEnv, "gains": 10.0},
    max_episode_steps=200,
)
gym.register(
    id="Pendulum_PC-v1",
    entry_point="action_space_toolbox:create_pc_env",
    kwargs={"base_env_type": PendulumEnv, "p_gains": 15.0, "d_gains": 2.0},
    max_episode_steps=200,
)
