import dataclasses
import pickle
from pathlib import Path
from typing import List, Optional

import filelock
import stable_baselines3
import torch
from stable_baselines3.common.type_aliases import RolloutBufferSamples

from action_space_toolbox.analysis.hessian.sb3_hessian import SB3Hessian
from action_space_toolbox.analysis.util import read_dict_recursive, write_dict_recursive


@dataclasses.dataclass
class ComputationParameters:
    tol: float
    max_iter: int
    num_samples: int


@dataclasses.dataclass
class HessianEigenResult:
    eigenvalues: List[float]
    eigenvectors: List[List[torch.Tensor]]
    computation_params: ComputationParameters


class HessianEigenCachedCalculator:
    def __init__(
        self,
        run_dir: Path,
        num_eigen: int = 50,
        tol: float = 1e-4,
        max_iter: int = 10000,
    ):
        self.cache_path = run_dir / "cashed_results" / "hessian_eigenvalues.pkl"
        self.cache_path.parent.mkdir(exist_ok=True)
        self.num_eigen = num_eigen
        self.tol = tol
        self.max_iter = max_iter

    def get_eigen(
        self,
        agent: stable_baselines3.ppo.PPO,
        data: RolloutBufferSamples,
        env_step: int,
        overwrite_cache: bool = False,
        show_progress: bool = False,
    ) -> HessianEigenResult:
        cached_evs = self.read_cached_eigen(env_step)
        if (
            not overwrite_cache
            and cached_evs is not None
            and len(cached_evs.eigenvalues) >= self.num_eigen
        ):
            return cached_evs
        else:
            hessian_comp = SB3Hessian(agent, data, agent.device)
            eigenvalues, eigenvectors = hessian_comp.eigenvalues(
                self.max_iter, self.tol, self.num_eigen, show_progress
            )
            # Sometimes the eigenvalues are not sorted properly, so sort eigenvalues and eigenvectors according to the
            # absolute value of the eigenvalues
            eigenvalues, eigenvectors = (
                list(t)
                for t in zip(
                    *sorted(
                        zip(eigenvalues, eigenvectors),
                        key=lambda x: abs(x[0]),
                        reverse=True,
                    )
                )
            )

            result = HessianEigenResult(
                eigenvalues,
                eigenvectors,
                ComputationParameters(self.tol, self.max_iter, data.actions.shape[0]),
            )
            self.cache_eigen(result, env_step)
            return result

    def read_cached_eigen(self, env_step: int) -> Optional[HessianEigenResult]:
        if not self.cache_path.exists():
            return None
        else:
            with filelock.FileLock(
                self.cache_path.with_suffix(self.cache_path.suffix + ".lock")
            ):
                with self.cache_path.open("rb") as cache_file:
                    cache = pickle.load(cache_file)
            cached_eigen = read_dict_recursive(cache, ("ppo", env_step))
            return cached_eigen

    def cache_eigen(self, result: HessianEigenResult, env_step: int) -> None:
        cache = self.read_cached_eigen(env_step)
        if cache is None:
            cache = {}
        with filelock.FileLock(
            self.cache_path.with_suffix(self.cache_path.suffix + ".lock")
        ):
            with self.cache_path.open("wb") as cache_file:
                write_dict_recursive(cache, ("ppo", env_step), result)
                pickle.dump(cache, cache_file)
