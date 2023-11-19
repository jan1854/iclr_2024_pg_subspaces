from typing import Dict, List, Tuple

import numpy as np
import stable_baselines3.common.vec_env.base_vec_env


def normalize(
    data: np.ndarray, mean: np.ndarray, std: np.ndarray, eps: float
) -> np.ndarray:
    return (data - mean) / (std + eps)


class FixedNormalizationVecWrapper(
    stable_baselines3.common.vec_env.base_vec_env.VecEnvWrapper
):
    def __init__(
        self,
        venv: stable_baselines3.common.vec_env.base_vec_env.VecEnv,
        mean: np.ndarray,
        std: np.ndarray,
        eps: float,
    ):
        super().__init__(venv)
        self.mean = mean
        self.std = std
        self.eps = eps

    def reset(self) -> np.ndarray:
        return normalize(self.venv.reset(), self.mean, self.std, self.eps)

    def step_wait(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
        observations, rewards, dones, infos = self.venv.step_wait()
        return (
            normalize(observations, self.mean, self.std, self.eps),
            rewards,
            dones,
            infos,
        )
