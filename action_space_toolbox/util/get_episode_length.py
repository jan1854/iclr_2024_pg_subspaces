import gym

from action_space_toolbox.control_modes.check_wrapped import check_wrapped


def get_episode_length(env: gym.Env) -> int:
    # This is quite an ugly hack but there is no elegant way to get the environment's time limit at the moment
    assert check_wrapped(env, gym.wrappers.TimeLimit)
    while not hasattr(env, "_max_episode_steps"):
        env = env.env
    return env._max_episode_steps
