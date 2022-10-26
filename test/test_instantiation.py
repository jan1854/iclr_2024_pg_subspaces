import re

import gym

from action_space_toolbox import BASE_ENV_TYPE_OR_ID

SUPPORTED_CONTROL_MODES = ["PC", "VC"]


def construct_env_id(base_env_id: str, control_mode_id: str) -> str:
    version_str = re.findall("-v[0-9]+", base_env_id)[-1]
    return f"{base_env_id[:-len(version_str)]}_{control_mode_id}{version_str}"


def test_instantiation():
    """
    Instantiates all implemented environments to test whether any exceptions occur during the instantiation.
    """
    for env_id in BASE_ENV_TYPE_OR_ID.keys():
        for control_mode_id in SUPPORTED_CONTROL_MODES:
            env = gym.make(construct_env_id(env_id, control_mode_id))

            env.reset()
            action = env.action_space.sample()
            env.step(action)
