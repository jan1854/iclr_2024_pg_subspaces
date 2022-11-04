import logging
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

from action_space_toolbox.controller_base.controller_base_wrapper import (
    ControllerBaseWrapper,
)
from action_space_toolbox.controller_base.dmc_controller_base_wrapper import (
    DMCControllerBaseWrapper,
)
from action_space_toolbox.controller_base.gym_mujoco_controller_base_wrapper import (
    GymMujocoControllerBaseWrapper,
)
from action_space_toolbox.controller_base.pendulum_controller_base_wrapper import (
    PendulumControllerBaseWrapper,
)
from action_space_toolbox.control_modes.position_control_wrapper import (
    PositionControlWrapper,
)
from action_space_toolbox.control_modes.velocity_control_wrapper import (
    VelocityControlWrapper,
)
from action_space_toolbox.reward.reacher_disable_control_reward_wrapper import (
    ReacherDisableControlRewardWrapper,
)
from action_space_toolbox.util.construct_env_id import construct_env_id
from action_space_toolbox.util.action_repeat_wrapper import ActionRepeatWrapper

TEnv = TypeVar("TEnv", bound=gym.Env)

logger = logging.getLogger(__name__)


def create_base_env(
    base_env_type_or_id: Union[Type[TEnv], Tuple[str, str]],
    disable_control_rewards: bool = False,
    **kwargs,
):
    # Action repeat is handled by the ActionRepeatWrapper (accessible with the action_repeat argument to gym.make())
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
        if base_env_type_or_id is ReacherEnv:
            base_env = base_env_type_or_id(**kwargs)
            if disable_control_rewards:
                base_env = ReacherDisableControlRewardWrapper(base_env)
        else:
            if disable_control_rewards:
                base_env = base_env_type_or_id(**kwargs, ctrl_cost_weight=0.0)
            else:
                base_env = base_env_type_or_id(**kwargs)
        return base_env


def wrap_env_controller_base(env: gym.Env) -> ControllerBaseWrapper:
    if isinstance(env.unwrapped, PendulumEnv):
        return PendulumControllerBaseWrapper(env)
    elif isinstance(env.unwrapped, MujocoEnv):
        return GymMujocoControllerBaseWrapper(env)
    elif isinstance(env.unwrapped, dmc2gym.wrappers.DMCWrapper):
        return DMCControllerBaseWrapper(env)
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


def add_common_wrappers(
    env: gym.Env, normalize: bool, action_repeat: int, max_episode_steps: int
) -> gym.Env:
    env = maybe_wrap_action_normalization(env, normalize)
    env = maybe_wrap_action_repeat(env, action_repeat)
    if max_episode_steps % env.base_env_timestep_factor != 0:
        logger.warning(
            f"The episode length of the original environment {max_episode_steps} is not divisible by the "
            f"action repeat {env.base_env_timestep_factor}. Episodes will be shorter than specified."
        )
    env = gym.wrappers.TimeLimit(env, max_episode_steps // env.base_env_timestep_factor)
    return env


def create_vc_env(
    base_env_type_or_id: Union[Type[TEnv], Tuple[str, str]],
    gains: np.ndarray,
    target_velocity_limits: Optional[Union[float, Sequence[float]]] = None,
    controller_steps: int = 1,
    keep_base_timestep: bool = True,
    normalize: bool = True,
    action_repeat: int = 1,
    max_episode_steps: int = 1000,
    disable_control_rewards: bool = False,
    **kwargs,
) -> gym.Env:
    env = create_base_env(base_env_type_or_id, disable_control_rewards, **kwargs)
    env = wrap_env_controller_base(env)
    env = VelocityControlWrapper(
        env, gains, target_velocity_limits, controller_steps, keep_base_timestep
    )
    return add_common_wrappers(env, normalize, action_repeat, max_episode_steps)


def create_pc_env(
    base_env_type_or_id: Union[Type[TEnv], Tuple[str, str]],
    p_gains: np.ndarray,
    d_gains: np.ndarray,
    target_position_limits: Optional[Union[float, Sequence[float]]] = None,
    positions_relative: bool = False,
    adaptive_relative_position_limits: bool = False,
    controller_steps: int = 1,
    keep_base_timestep: bool = True,
    normalize: bool = True,
    action_repeat: int = 1,
    max_episode_steps: int = 1000,
    disable_control_rewards: bool = False,
    **kwargs,
) -> gym.Env:
    env = create_base_env(base_env_type_or_id, disable_control_rewards, **kwargs)
    env = wrap_env_controller_base(env)
    env = PositionControlWrapper(
        env,
        p_gains,
        d_gains,
        positions_relative,
        adaptive_relative_position_limits,
        target_position_limits,
        controller_steps,
        keep_base_timestep,
    )
    return add_common_wrappers(env, normalize, action_repeat, max_episode_steps)


def create_tc_env(
    base_env_type_or_id: Union[Type[TEnv], Tuple[str, str]],
    normalize: bool = True,
    action_repeat: int = 1,
    max_episode_steps: int = 1000,
    disable_control_rewards: bool = False,
    **kwargs,
) -> gym.Env:
    env = create_base_env(base_env_type_or_id, disable_control_rewards, **kwargs)
    env = wrap_env_controller_base(env)
    return add_common_wrappers(env, normalize, action_repeat, max_episode_steps)


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

res_path = Path(__file__).parent / "res"
pc_parameters_path = res_path / "pc_parameters.yaml"
with pc_parameters_path.open("r") as pc_parameters_file:
    pc_parameters = yaml.safe_load(pc_parameters_file)
vc_parameters_path = res_path / "vc_parameters.yaml"
with vc_parameters_path.open("r") as vc_parameters_file:
    vc_parameters = yaml.safe_load(vc_parameters_file)
original_env_args_path = res_path / "original_env_args.yaml"
with original_env_args_path.open("r") as original_env_args_file:
    original_env_args = yaml.safe_load(original_env_args_file)
custom_env_args_path = res_path / "custom_env_args.yaml"
with custom_env_args_path.open("r") as custom_env_args_file:
    custom_env_args = yaml.safe_load(custom_env_args_file)

control_mode_parameters = {"TC": {}, "VC": vc_parameters, "PC": pc_parameters}
ENTRY_POINTS = {
    "TC": "action_space_toolbox:create_tc_env",
    "VC": "action_space_toolbox:create_vc_env",
    "PC": "action_space_toolbox:create_pc_env",
}

for base_env_name, base_env_type_or_id in BASE_ENV_TYPE_OR_ID.items():
    for control_mode in control_mode_parameters:
        curr_control_mode_parameters = control_mode_parameters[control_mode].get(
            base_env_name, {}
        )
        curr_custom_env_args = custom_env_args.get(base_env_name, {})
        parameters = (
            DEFAULT_PARAMETERS[control_mode]
            | curr_control_mode_parameters
            | curr_custom_env_args
        )
        env_args = original_env_args.get(base_env_name, {})

        gym.register(
            id=construct_env_id(base_env_name, control_mode),
            entry_point=ENTRY_POINTS[control_mode],
            kwargs={"base_env_type_or_id": base_env_type_or_id, **parameters},
            **env_args,
        )
