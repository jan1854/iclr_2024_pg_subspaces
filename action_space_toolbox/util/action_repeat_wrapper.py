import logging
from typing import Dict, Tuple

import gym
import numpy as np


logger = logging.getLogger(__name__)


class ActionRepeatWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, frame_skip: int):
        super().__init__(env)
        assert frame_skip >= 1
        self.frame_skip = frame_skip
        if (
            getattr(self.env, "_max_episode_steps")
            and self.env._max_episode_steps % frame_skip != 0
        ):
            logger.warning(
                f"The episode length {self.env._max_episode_steps} is not divisible by the frame skip "
                f"{frame_skip}. Episodes will end with incomplete steps."
            )

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        total_reward = 0.0
        for _ in range(self.frame_skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info
