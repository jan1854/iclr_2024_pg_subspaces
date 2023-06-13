import re
from pathlib import Path
from typing import Literal, Optional, Sequence, Tuple, Union

import filelock
import numpy as np
import stable_baselines3
import torch
from stable_baselines3.common.type_aliases import RolloutBufferSamples

from action_space_toolbox.analysis.hessian.calculate_hessian import calculate_hessian
from action_space_toolbox.analysis.util import flatten_parameters
from action_space_toolbox.util.sb3_training import ppo_loss


def _get_cache_paths(
    cache_path: Path, env_step: int, num_grad_steps_additional_training: int
) -> Tuple[Path, Path]:
    paths_without_suffix = (
        cache_path / f"eigenvalues_{env_step:07d}",
        cache_path / f"eigenvectors_{env_step:07d}",
    )
    if num_grad_steps_additional_training > 0:
        paths_without_suffix = tuple(
            p.with_name(
                p.name
                + f"_additional_grad_steps_{num_grad_steps_additional_training:05d}"
            )
            for p in paths_without_suffix
        )
    return tuple(p.with_suffix(".npy") for p in paths_without_suffix)  # type: ignore


class CachedEigenIterator:
    def __init__(
        self,
        cache_path: Path,
        env_steps: Sequence[int],
        num_grad_steps_additional_training: int,
        device: Union[str, torch.device],
    ):
        self.cache_path = cache_path
        self.env_steps = tuple(env_steps)
        self.num_grad_steps_additional_training = num_grad_steps_additional_training
        self.device = device
        self._idx = 0

    def __iter__(self) -> "CachedEigenIterator":
        return self

    def __next__(self) -> Tuple[int, torch.Tensor, torch.Tensor]:
        if self._idx < len(self.env_steps):
            env_step = self.env_steps[self._idx]
            eigenvals_path, eigenvecs_path = _get_cache_paths(
                self.cache_path, env_step, self.num_grad_steps_additional_training
            )
            eigenvals = torch.tensor(np.load(str(eigenvals_path)), device=self.device)
            eigenvecs = torch.tensor(np.load(str(eigenvecs_path)), device=self.device)
            self._idx += 1
            return env_step, eigenvals, eigenvecs
        else:
            raise StopIteration


class HessianEigenCachedCalculator:
    def __init__(
        self,
        run_dir: Path,
        num_eigenvectors_to_cache: int = 200,
        device: Union[str, torch.device] = "cpu",
    ):
        self.cache_path = run_dir / "cached_results" / "eigen"
        self.cache_path.mkdir(exist_ok=True, parents=True)
        self.num_eigenvectors_to_cache = num_eigenvectors_to_cache
        self.device = device

    def get_eigen(
        self,
        agent: stable_baselines3.ppo.PPO,
        data: RolloutBufferSamples,
        env_step: int,
        num_grad_steps_additional_training: int = 0,
        num_eigenvectors: Union[int, Literal["all"], None] = None,
        overwrite_cache: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if num_eigenvectors == "all":
            num_eigenvectors = len(flatten_parameters(agent.policy.parameters()))
        cached_eigen = self.read_cached_eigen(
            env_step, num_grad_steps_additional_training
        )
        if (
            not overwrite_cache
            and cached_eigen is not None
            and (
                num_eigenvectors is None or cached_eigen[1].shape[0] >= num_eigenvectors
            )
            # Legacy: Make sure that the eigenvalues are sorted in descending order (to make sure that the eigenvectors
            #         w.r.t. the largest eigenvalues are cached)
            and torch.all(cached_eigen[0][:-1] > cached_eigen[0][1:])
        ):
            eigenvalues, eigenvectors = cached_eigen
        else:

            hess = calculate_hessian(agent, lambda a: ppo_loss(a, data)[0])
            eigenvalues, eigenvectors = torch.linalg.eigh(hess)
            indices_sorted = eigenvalues.argsort(descending=True)
            eigenvalues = eigenvalues[indices_sorted]
            eigenvectors = eigenvectors[:, indices_sorted]
            self.cache_eigen(
                eigenvalues, eigenvectors, env_step, num_grad_steps_additional_training
            )
        if num_eigenvectors is not None:
            eigenvectors = eigenvectors[:, :num_eigenvectors]
        return eigenvalues, eigenvectors

    def read_cached_eigen(
        self, env_step: int, num_grad_steps_additional_training: int
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        eigenval_cache_path, eigenvec_cache_path = _get_cache_paths(
            self.cache_path, env_step, num_grad_steps_additional_training
        )
        if not eigenval_cache_path.exists():
            return None
        else:
            with filelock.FileLock(
                eigenval_cache_path.with_suffix(eigenval_cache_path.suffix + ".lock")
            ):
                eigenvalues = np.load(str(eigenval_cache_path))
                eigenvectors = np.load(str(eigenvec_cache_path))
            return torch.tensor(eigenvalues, device=self.device), torch.tensor(
                eigenvectors, device=self.device
            )

    def cache_eigen(
        self,
        eigenvalues: torch.Tensor,
        eigenvectors: torch.Tensor,
        env_step: int,
        num_grad_steps_additional_training: int,
    ) -> None:
        eigenval_cache_path, eigenvec_cache_path = _get_cache_paths(
            self.cache_path, env_step, num_grad_steps_additional_training
        )
        with filelock.FileLock(
            eigenval_cache_path.with_suffix(eigenval_cache_path.suffix + ".lock")
        ):
            np.save(str(eigenval_cache_path), eigenvalues.cpu().numpy())
            np.save(
                str(eigenvec_cache_path),
                eigenvectors[:, : self.num_eigenvectors_to_cache].cpu().numpy(),
            )

    def iter_cached_eigen(
        self, num_grad_steps_additional_training: int = 0
    ) -> "CachedEigenIterator":
        if num_grad_steps_additional_training == 0:
            pattern = "eigenvalues_[0-9]+.npy"
        else:
            pattern = f"eigenvalues_[0-9]+_additional_grad_steps_{num_grad_steps_additional_training:05d}.npy"
        env_steps = [
            # TODO: Does not work if the number of env_steps is > 10000000 --> Use re again
            int(cache_file.name[len("eigenvalues") + 1 : len("eigenvalues") + 8])
            for cache_file in self.cache_path.iterdir()
            if re.fullmatch(pattern, cache_file.name)
        ]
        return CachedEigenIterator(
            self.cache_path,
            sorted(env_steps),
            num_grad_steps_additional_training,
            self.device,
        )
