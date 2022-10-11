import gym.envs.classic_control

from action_space_toolbox.base_environments.pendulum_controller_env import (
    PendulumControllerEnv,
)
from action_space_toolbox.base_environments.controller_base_env import ControllerBaseEnv
from action_space_toolbox.control_modes.position_control_wrapper import (
    PositionControlWrapper,
)
from action_space_toolbox.control_modes.velocity_control_wrapper import (
    VelocityControlWrapper,
)


def create_vc_env(env_type, gains, **kwargs):
    return VelocityControlWrapper(env_type(**kwargs), gains)


def create_pc_env(env_type, p_gains, d_gains, **kwargs):
    return PositionControlWrapper(env_type(**kwargs), p_gains, d_gains)


gym.register(
    id="Pendulum_VC-v1",
    entry_point="action_space_toolbox:create_vc_env",
    kwargs={"env_type": PendulumControllerEnv, "gains": 10.0},
    max_episode_steps=200,
)
gym.register(
    id="Pendulum_PC-v1",
    entry_point="action_space_toolbox:create_pc_env",
    kwargs={"env_type": PendulumControllerEnv, "p_gains": 15.0, "d_gains": 2.0},
    max_episode_steps=200,
)
