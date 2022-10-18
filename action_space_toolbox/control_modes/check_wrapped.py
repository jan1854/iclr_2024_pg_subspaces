from typing import TypeVar, Type

import gym


TWrapper = TypeVar("TWrapper", bound=gym.Wrapper)


def check_wrapped(env: gym.Env, wrapper: Type[TWrapper]):
    """
    Checks whether a given environment is wrapped in a JointInformationWrapper.

    :param env:     The gym environment to check
    :return:        True iff the environment is wrapped in a JointInformationWrapper
    """
    while isinstance(env, gym.Wrapper):
        if isinstance(env, wrapper):
            return True
        env = env.env
    return False
