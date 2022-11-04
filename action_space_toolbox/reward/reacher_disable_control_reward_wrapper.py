from typing import Tuple, Dict

import gym
import numpy as np
from gym.envs.mujoco import ReacherEnv


class ReacherDisableControlRewardWrapper(gym.Wrapper):
    def __int__(self, env: gym.Env):
        assert isinstance(env.unwrapped, ReacherEnv)
        super().__init__(env)

    def step(self, action: np.array) -> Tuple[np.ndarray, float, bool, Dict]:
        obs, _, done, info = self.env.step(action)
        return obs, info["reward_dist"], done, info
