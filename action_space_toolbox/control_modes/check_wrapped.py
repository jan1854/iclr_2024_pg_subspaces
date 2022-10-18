from typing import TypeVar, Type

import gym


TWrapper = TypeVar("TWrapper", bound=gym.Wrapper)


def check_wrapped(env: gym.Env, wrapper: Type[TWrapper]) -> bool:
    """
    Checks whether a given environment is wrapped with a given wrapper.

    :param env:     The gym environment to check
    :param wrapper: The wrapper to check for
    :return:        True iff the environment is wrapped in the given wrapper
    """
    while isinstance(env, gym.Wrapper):
        if isinstance(env, wrapper):
            return True
        env = env.env
    return False
