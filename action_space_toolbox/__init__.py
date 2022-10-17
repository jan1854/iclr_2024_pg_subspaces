from typing import Type, TypeVar, Union, Sequence, Optional

import gym.envs.classic_control
import numpy as np
from gym.envs.classic_control import PendulumEnv
from gym.envs.mujoco import MujocoEnv
from gym.envs.mujoco.ant_v3 import AntEnv
from gym.envs.mujoco.half_cheetah_v3 import HalfCheetahEnv
from gym.envs.mujoco.hopper_v3 import HopperEnv
from gym.envs.mujoco.walker2d_v3 import Walker2dEnv

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

ORIGINAL_ENV_ARGS = {
    # Classic control
    "Pendulum-v1": {"max_episode_steps": 200},
    # MuJoCo
    "Ant-v3": {"max_episode_steps": 1000, "reward_threshold": 6000.0},
    "HalfCheetah-v3": {"max_episode_steps": 1000, "reward_threshold": 4800.0},
    "Hopper-v3": {"max_episode_steps": 1000, "reward_threshold": 3800.0},
    "Walker2d-v3": {"max_episode_steps": 1000},
}


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
    base_env_type: Type[TEnv],
    gains: np.ndarray,
    target_velocity_limits: Optional[Union[float, Sequence[float]]] = None,
    **kwargs,
) -> VelocityControlWrapper:
    return VelocityControlWrapper(
        wrap_env_dof_information(base_env_type(**kwargs)), gains, target_velocity_limits
    )


def create_pc_env(
    base_env_type: Type[TEnv],
    p_gains: np.ndarray,
    d_gains: np.ndarray,
    positions_relative: bool,
    target_position_limits: Optional[Union[float, Sequence[float]]] = None,
    **kwargs,
) -> PositionControlWrapper:
    return PositionControlWrapper(
        wrap_env_dof_information(base_env_type(**kwargs)),
        p_gains,
        d_gains,
        positions_relative,
        target_position_limits,
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
    kwargs={
        "base_env_type": PendulumEnv,
        "p_gains": 15.0,
        "d_gains": 2.0,
        "positions_relative": False,
    },
    max_episode_steps=200,
)

# TODO: Need to tune the controller gains and the velocity limits of all environments below
gym.register(
    id="Ant_VC-v3",
    entry_point="action_space_toolbox:create_vc_env",
    kwargs={
        "base_env_type": AntEnv,
        "gains": 10.0,
        "target_velocity_limits": [-10.0, 10.0],
    },
    **ORIGINAL_ENV_ARGS["Ant-v3"],
)
gym.register(
    id="Ant_PC-v3",
    entry_point="action_space_toolbox:create_pc_env",
    kwargs={
        "base_env_type": AntEnv,
        "p_gains": 15.0,
        "d_gains": 2.0,
        "positions_relative": False,
    },
    **ORIGINAL_ENV_ARGS["Ant-v3"],
)

gym.register(
    id="HalfCheetah_VC-v3",
    entry_point="action_space_toolbox:create_vc_env",
    kwargs={
        "base_env_type": HalfCheetahEnv,
        "gains": 10.0,
        "target_velocity_limits": [-10.0, 10.0],
    },
    **ORIGINAL_ENV_ARGS["HalfCheetah-v3"],
)
gym.register(
    id="HalfCheetah_PC-v3",
    entry_point="action_space_toolbox:create_pc_env",
    kwargs={
        "base_env_type": HalfCheetahEnv,
        "p_gains": 15.0,
        "d_gains": 2.0,
        "positions_relative": False,
    },
    **ORIGINAL_ENV_ARGS["HalfCheetah-v3"],
)

gym.register(
    id="Hopper_VC-v3",
    entry_point="action_space_toolbox:create_vc_env",
    kwargs={
        "base_env_type": HopperEnv,
        "gains": 10.0,
        "target_velocity_limits": [-10.0, 10.0],
    },
    **ORIGINAL_ENV_ARGS["Hopper-v3"],
)
gym.register(
    id="Hopper_PC-v3",
    entry_point="action_space_toolbox:create_pc_env",
    kwargs={
        "base_env_type": HopperEnv,
        "p_gains": 15.0,
        "d_gains": 2.0,
        "positions_relative": False,
    },
    **ORIGINAL_ENV_ARGS["Hopper-v3"],
)

gym.register(
    id="Walker2d_VC-v3",
    entry_point="action_space_toolbox:create_vc_env",
    kwargs={
        "base_env_type": Walker2dEnv,
        "gains": 10.0,
        "target_velocity_limits": [-10.0, 10.0],
    },
    **ORIGINAL_ENV_ARGS["Walker2d-v3"],
)
gym.register(
    id="Walker2d_PC-v3",
    entry_point="action_space_toolbox:create_pc_env",
    kwargs={
        "base_env_type": Walker2dEnv,
        "p_gains": 15.0,
        "d_gains": 2.0,
        "positions_relative": False,
    },
    **ORIGINAL_ENV_ARGS["Walker2d-v3"],
)
