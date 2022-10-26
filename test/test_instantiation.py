import gym

from action_space_toolbox import BASE_ENV_TYPE_OR_ID, construct_env_id, ENTRY_POINTS


def test_instantiation():
    """
    Instantiates all implemented environments to test whether any exceptions occur during the instantiation.
    """
    for env_id in BASE_ENV_TYPE_OR_ID.keys():
        for control_mode in ENTRY_POINTS.keys():
            env = gym.make(construct_env_id(env_id, control_mode))

            env.reset()
            action = env.action_space.sample()
            env.step(action)
