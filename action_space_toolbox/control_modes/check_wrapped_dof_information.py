import gym

from action_space_toolbox.dof_information.dof_information_wrapper import (
    DofInformationWrapper,
)


def check_wrapped_dof_information(env: gym.Env):
    """
    Checks whether a given environment is wrapped in a JointInformationWrapper.

    :param env:     The gym environment to check
    :return:        True iff the environment is wrapped in a JointInformationWrapper
    """
    while isinstance(env, gym.Wrapper):
        if isinstance(env, DofInformationWrapper):
            return True
        env = env.env
    return False
