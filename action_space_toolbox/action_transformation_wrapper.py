import abc
from typing import Dict, Tuple

import gym
import numpy as np


class ActionTransformationWrapper(gym.Wrapper, abc.ABC):
    """
    An action transformation wrapper similar to gym.core.ActionWrapper. The main difference is that this wrapper allows
    for state-full transformations (requires changing the observations).
    """

    def __init__(self, env: gym.Env, repeat: int = 1, keep_base_timestep: bool = True):
        """
        :param env:                 The gym environment to wrap
        :param repeat:              Number of steps for which the wrapped environment should be executed for each step
                                    of the wrapper (rewards are averaged and the last observation of the wrapped
                                    environment is used as observation of the wrapper)
        :param keep_base_timestep:  Whether the simulation timestep of the base environment should be kept. If false,
                                    the timestep of the base environment will be changed to original timestep / repeat
                                    (so that the resulting time of one step of the wrapper is the same as the original
                                    timestep)
        """
        super().__init__(env)
        assert repeat >= 1
        self.repeat = repeat
        self.keep_base_timestep = keep_base_timestep
        if not keep_base_timestep:
            self.env.set_timestep(self.env.timestep / repeat)

    def reset(self, **kwargs) -> np.ndarray:
        self.reset_transformation()
        return self.transform_state(self.env.reset())

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        rewards = []
        for _ in range(self.repeat):
            action_transformed = self.transform_action(action)
            action_transformed = np.core.umath.clip(  # For some reason np.core.umath.clip is a lot faster than np.clip
                action_transformed,
                self.env.action_space.low,
                self.env.action_space.high,
            )
            obs, reward, done, info = self.env.step(action_transformed)
            rewards.append(reward)
            if done:
                break
        if self.keep_base_timestep:
            reward = np.sum(rewards)
        else:
            reward = np.mean(rewards)
        return self.transform_state(obs), reward, done, info

    def reset_transformation(self) -> None:
        return

    def transformation_observation(self) -> np.ndarray:
        return np.array([])

    def transform_state(self, state: np.ndarray) -> np.ndarray:
        return np.concatenate((state, self.transformation_observation()))

    @abc.abstractmethod
    def transform_action(self, action: np.ndarray) -> np.ndarray:
        pass

    @property
    def base_env_timestep_factor(self):
        if self.keep_base_timestep:
            return self.env.base_env_timestep_factor * self.repeat
        else:
            return self.env.base_env_timestep_factor
