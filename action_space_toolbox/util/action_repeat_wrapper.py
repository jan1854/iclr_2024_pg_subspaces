import logging
from typing import Dict, Tuple

import gym
import numpy as np


logger = logging.getLogger(__name__)


class ActionRepeatWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, action_repeat: int):
        super().__init__(env)
        assert action_repeat >= 1
        self.action_repeat = action_repeat

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        total_reward = 0.0
        for _ in range(self.action_repeat):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info

    @property
    def base_env_timestep_factor(self):
        return self.env.base_env_timestep_factor * self.action_repeat
