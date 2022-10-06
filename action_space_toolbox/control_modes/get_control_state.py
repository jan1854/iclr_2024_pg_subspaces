from typing import Tuple

import gym.envs.classic_control
import numpy as np


def get_control_state(env: gym.Env) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts the current state (position and velocity) of all controllable components (e.g. joints) of the given
    environment.

    @param env:     the gym environment to extract the control state of
    @return:        a (position, velocity)-tuple, where position and velocity are arrays containing the value for each
                    controllable component
    """
    if isinstance(env.unwrapped, gym.envs.classic_control.PendulumEnv):
        return np.array([env.state[0]]), np.array([env.state[1]])
    else:
        raise NotImplementedError(f"get_control_state does not support environments of type {type(env.unwrapped)}.")
