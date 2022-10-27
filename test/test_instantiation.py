import gym
import numpy as np

import action_space_toolbox
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

            assert np.all(env.action_space.low == -1) and np.all(env.action_space.high == 1)
            assert env.action_space.shape == env.unwrapped.action_space.shape


def test_unused_parameters():
    """
    Check that all parameters defined in pc_parameters.yaml / vc_parameters.yaml belong to a supported environment.
    """
    for parameters in action_space_toolbox.control_mode_parameters.values():
        for env_name in parameters.keys():
            assert env_name in BASE_ENV_TYPE_OR_ID
