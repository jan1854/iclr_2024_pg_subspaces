from pathlib import Path
from typing import Type, TypeVar, Union, Sequence, Optional, Tuple

import dm_control.suite
import dmc2gym
import dmc2gym.wrappers
import gym.envs.classic_control
import numpy as np
import yaml
from gym.envs.classic_control import PendulumEnv
from gym.envs.mujoco import MujocoEnv, ReacherEnv
from gym.envs.mujoco.ant_v3 import AntEnv
from gym.envs.mujoco.half_cheetah_v3 import HalfCheetahEnv
from gym.envs.mujoco.hopper_v3 import HopperEnv
from gym.envs.mujoco.walker2d_v3 import Walker2dEnv

from action_space_toolbox.dof_information.dof_information_wrapper import (
    DofInformationWrapper,
)
from action_space_toolbox.dof_information.dmc_dof_information_wrapper import (
    DMCDofInformationWrapper,
)
from action_space_toolbox.dof_information.gym_mujoco_dof_information_wrapper import (
    GymMujocoDofInformationWrapper,
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
from action_space_toolbox.util.construct_env_id import construct_env_id
from action_space_toolbox.util.action_repeat_wrapper import ActionRepeatWrapper

TEnv = TypeVar("TEnv", bound=gym.Env)


def create_base_env(base_env_type_or_id: Union[Type[TEnv], Tuple[str, str]], **kwargs):
    assert "frame_skip" not in kwargs or kwargs["frame_skip"] == 1
    # The step limit of the environment is problematic when using a different controller frequency, therefore we disable
    # it / set a very large value and handle termination with a TimeLimitWrapper later
    if isinstance(base_env_type_or_id, tuple):
        time_limit_wrapped_env = dmc2gym.make(
            base_env_type_or_id[0],
            base_env_type_or_id[1],
            time_limit=int(1e9),
            height=480,
            width=640,
            **kwargs,
        )
        # Remove the TimeLimitWrapper added by dmc2gym (see above)
        return time_limit_wrapped_env.env
    else:
        return base_env_type_or_id(**kwargs)


def wrap_env_dof_information(env: gym.Env) -> DofInformationWrapper:
    if isinstance(env.unwrapped, PendulumEnv):
        return PendulumDofInformationWrapper(env)
    elif isinstance(env.unwrapped, MujocoEnv):
        return GymMujocoDofInformationWrapper(env)
    elif isinstance(env.unwrapped, dmc2gym.wrappers.DMCWrapper):
        return DMCDofInformationWrapper(env)
    else:
        raise NotImplementedError(
            f"Environment {type(env.unwrapped)} is not supported."
        )


def maybe_wrap_action_normalization(env: gym.Env, normalize: bool) -> gym.Env:
    if (
        not normalize
        or np.all(env.action_space.low == -1)
        and np.all(env.action_space.high == 1)
    ):
        return env
    else:
        return gym.wrappers.RescaleAction(env, -1, 1)


def maybe_wrap_action_repeat(env: gym.Env, action_repeat: int) -> gym.Env:
    if action_repeat > 1:
        return ActionRepeatWrapper(env, action_repeat)
    else:
        return env


def create_vc_env(
    base_env_type_or_id: Union[Type[TEnv], Tuple[str, str]],
    gains: np.ndarray,
    target_velocity_limits: Optional[Union[float, Sequence[float]]] = None,
    controller_steps: int = 1,
    normalize: bool = True,
    action_repeat: int = 1,
    **kwargs,
) -> gym.Env:
    base_env = create_base_env(base_env_type_or_id, **kwargs)
    env = VelocityControlWrapper(
        wrap_env_dof_information(base_env),
        gains,
        target_velocity_limits,
        controller_steps,
    )
    env = maybe_wrap_action_normalization(env, normalize)
    return maybe_wrap_action_repeat(env, action_repeat)


def create_pc_env(
    base_env_type_or_id: Union[Type[TEnv], Tuple[str, str]],
    p_gains: np.ndarray,
    d_gains: np.ndarray,
    target_position_limits: Optional[Union[float, Sequence[float]]] = None,
    controller_steps: int = 1,
    normalize: bool = True,
    action_repeat: int = 1,
    **kwargs,
) -> gym.Env:
    base_env = create_base_env(base_env_type_or_id, **kwargs)
    env = PositionControlWrapper(
        wrap_env_dof_information(base_env),
        p_gains,
        d_gains,
        target_position_limits=target_position_limits,
        controller_steps=controller_steps,
    )
    env = maybe_wrap_action_normalization(env, normalize)
    return maybe_wrap_action_repeat(env, action_repeat)


def create_tc_env(
    base_env_type_or_id: Union[Type[TEnv], Tuple[str, str]],
    normalize: bool = True,
    action_repeat: int = 1,
    **kwargs,
) -> gym.Env:
    base_env = create_base_env(base_env_type_or_id, **kwargs)
    env = maybe_wrap_action_normalization(base_env, normalize)
    return maybe_wrap_action_repeat(env, action_repeat)


# TODO: Support also fish and ball_in_cup tasks (find sensible limits for the positions)
BASE_ENV_TYPE_OR_ID = {
    # Classic control
    "Pendulum-v1": PendulumEnv,
    # MuJoCo
    "Ant-v3": AntEnv,
    "HalfCheetah-v3": HalfCheetahEnv,
    "Hopper-v3": HopperEnv,
    "Reacher-v2": ReacherEnv,
    "Walker2d-v3": Walker2dEnv,
} | {
    f"dmc_{domain.capitalize()}-{task}-v1": (domain, task)
    for domain, task in dm_control.suite.BENCHMARKING
    if domain != "fish" and domain != "ball_in_cup"
}

DEFAULT_PARAMETERS = {
    "TC": {},
    "VC": {"gains": 10.0},
    "PC": {"p_gains": 15.0, "d_gains": 2.0},
}

pc_parameters_path = Path(__file__).parent / "res" / "pc_parameters.yaml"
with pc_parameters_path.open("r") as pc_parameters_file:
    pc_parameters = yaml.safe_load(pc_parameters_file)
vc_parameters_path = Path(__file__).parent / "res" / "vc_parameters.yaml"
with vc_parameters_path.open("r") as vc_parameters_file:
    vc_parameters = yaml.safe_load(vc_parameters_file)
original_env_args_path = Path(__file__).parent / "res" / "original_env_args.yaml"
with original_env_args_path.open("r") as original_env_args_file:
    original_env_args = yaml.safe_load(original_env_args_file)

control_mode_parameters = {"TC": {}, "VC": vc_parameters, "PC": pc_parameters}
ENTRY_POINTS = {
    "TC": "action_space_toolbox:create_tc_env",
    "VC": "action_space_toolbox:create_vc_env",
    "PC": "action_space_toolbox:create_pc_env",
}

for base_env_name, base_env_type_or_id in BASE_ENV_TYPE_OR_ID.items():
    for control_mode in control_mode_parameters:
        parameters = DEFAULT_PARAMETERS[control_mode] | control_mode_parameters[
            control_mode
        ].get(base_env_name, {})
        env_args = original_env_args.get(base_env_name, {"max_episode_steps": 1000})

        gym.register(
            id=construct_env_id(base_env_name, control_mode),
            entry_point=ENTRY_POINTS[control_mode],
            kwargs={"base_env_type_or_id": base_env_type_or_id, **parameters},
            **env_args,
        )
