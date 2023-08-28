import gym
import numpy as np
import pytest

from action_space_toolbox.transformations.stateless.action_duplication_wrapper import ActionDuplicationWrapper


class IdentityEnv(gym.Env):
    def __init__(self):
        self.action_space = gym.spaces.Box(low=np.array([-1, -2, -3], dtype=np.float32), high=np.array([4, 5, 7], dtype=np.float32))

    def step(self, action):
        return action, 0.0, False, {}

    def reset(self):
        return np.zeros(3)

    def render(self, mode="human"):
        raise NotImplementedError()


def test_action_duplication_wrapper():
    env = ActionDuplicationWrapper(IdentityEnv(), 4)
    env.reset()
    action = np.array([0, -1, 0.5, 0.8, -2, -1, 0, -1, 1, 2, 3, 4], dtype=np.float32)
    state, _, _, _ = env.step(action)
    assert state == pytest.approx(np.array([0.075, -1, 2.5]))

    env_weighted = ActionDuplicationWrapper(IdentityEnv(), 4, weights=(0, 1, 2, 1))
    env_weighted.reset()
    state, _, _, _ = env_weighted.step(action)
    assert state == pytest.approx(np.array([0.2, -0.5, 3]))

    env_not_rescaled = ActionDuplicationWrapper(IdentityEnv(), 4, rescale_weights=False)
    env_not_rescaled.reset()
    state, _, _, _ = env_not_rescaled.step(action)
    assert state == pytest.approx(np.array([0.3, -2, 7]))
