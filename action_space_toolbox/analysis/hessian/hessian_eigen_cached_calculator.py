from pathlib import Path
from typing import Literal, Optional, Tuple, Union

import filelock
import stable_baselines3
import torch
from stable_baselines3.common.type_aliases import RolloutBufferSamples

from action_space_toolbox.analysis.hessian.calculate_hessian import calculate_hessian
from action_space_toolbox.analysis.util import flatten_parameters
from action_space_toolbox.util.sb3_training import ppo_loss


class HessianEigenCachedCalculator:
    def __init__(
        self,
        run_dir: Path,
        num_eigenvectors_to_cache: int = 100,
    ):
        self.cache_path = run_dir / "cached_results" / "eigen"
        self.cache_path.mkdir(exist_ok=True, parents=True)
        self.num_eigenvectors_to_cache = num_eigenvectors_to_cache

    def get_eigen(
        self,
        agent: stable_baselines3.ppo.PPO,
        data: RolloutBufferSamples,
        env_step: int,
        num_eigenvectors: Union[int, Literal["all"], None],
        overwrite_cache: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if num_eigenvectors == "all":
            num_eigenvectors = len(flatten_parameters(agent.policy.parameters()))
        elif num_eigenvectors is None:
            num_eigenvectors = 0
        cached_eigen = self.read_cached_eigen(env_step)
        if (
            not overwrite_cache
            and cached_eigen is not None
            and cached_eigen[1].shape[0] >= num_eigenvectors
        ):
            return cached_eigen
        else:
            hess = calculate_hessian(agent, lambda a: ppo_loss(a, data)[0])
            eigenvalues, eigenvectors = torch.linalg.eigh(hess)
            self.cache_eigen(eigenvalues, eigenvectors, env_step)
            return eigenvalues, eigenvectors

    def read_cached_eigen(
        self, env_step: int
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        eigenval_cache_path, eigenvec_cache_path = self._get_cache_paths(env_step)
        if not eigenval_cache_path.exists():
            return None
        else:
            with filelock.FileLock(
                eigenval_cache_path.with_suffix(eigenval_cache_path.suffix + ".lock")
            ):
                eigenvalues = torch.load(eigenval_cache_path)
                eigenvectors = torch.load(eigenvec_cache_path)
            return eigenvalues, eigenvectors

    def cache_eigen(
        self, eigenvalues: torch.Tensor, eigenvectors: torch.Tensor, env_step: int
    ) -> None:
        eigenval_cache_path, eigenvec_cache_path = self._get_cache_paths(env_step)
        with filelock.FileLock(
            eigenval_cache_path.with_suffix(eigenval_cache_path.suffix + ".lock")
        ):
            torch.save(eigenvalues, eigenval_cache_path)
            torch.save(
                eigenvectors[:, : self.num_eigenvectors_to_cache], eigenvec_cache_path
            )

    def _get_cache_paths(self, env_step: int) -> Tuple[Path, Path]:
        return (
            self.cache_path / f"eigenvalues_{env_step:07d}.pt",
            self.cache_path / f"eigenvectors_{env_step:07d}.pt",
        )
