from typing import Union

import gym
import stable_baselines3.common.vec_env

from action_space_toolbox.control_modes.check_wrapped import check_wrapped


def get_episode_length(
    env: Union[gym.Env, stable_baselines3.common.vec_env.VecEnv]
) -> int:
    # This is quite an ugly hack but there is no elegant way to get the environment's time limit at the moment
    if isinstance(env, stable_baselines3.common.vec_env.VecEnv):
        assert len(env.envs) == 1
        env = env.envs[0]
    assert check_wrapped(env, gym.wrappers.TimeLimit)
    while not hasattr(env, "_max_episode_steps"):
        env = env.env
    return env._max_episode_steps
