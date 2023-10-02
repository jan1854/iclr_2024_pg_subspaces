import abc
import dataclasses
from pathlib import Path
from typing import Literal, Tuple, Union

import numpy as np
import pg_subspaces.sb3_utils.common.parameters
import stable_baselines3.common.base_class
import torch
from stable_baselines3.common.type_aliases import (
    RolloutBufferSamples,
    ReplayBufferSamples,
)


@dataclasses.dataclass
class Eigen:
    eigenvalues: torch.Tensor
    eigenvectors: torch.Tensor


@dataclasses.dataclass
class ActorCriticEigen:
    policy: "Eigen"
    value_function: "Eigen"

    def dump(
        self, path: Path, num_eigenvectors_to_store: Union[Literal["all"], int]
    ) -> None:
        if num_eigenvectors_to_store == "all":
            eigenvecs_policy = self.policy.eigenvectors
            eigenvecs_vf = self.value_function.eigenvectors
        else:
            eigenvecs_policy = self.policy.eigenvectors[:, :num_eigenvectors_to_store]
            eigenvecs_vf = self.value_function.eigenvectors[
                :, :num_eigenvectors_to_store
            ]
        np.savez_compressed(
            str(path),
            **{
                "policy.eigenvalues": self.policy.eigenvalues.cpu().numpy(),
                "policy.eigenvectors": eigenvecs_policy.cpu().numpy(),
                "value_function.eigenvalues": self.value_function.eigenvalues.cpu().numpy(),
                "value_function.eigenvectors": eigenvecs_vf.cpu().numpy(),
            },
        )

    @classmethod
    def load(cls, path: Path, device: Union[str, torch.device]) -> "ActorCriticEigen":
        eigen_npz = np.load(str(path), allow_pickle=True)
        eigen_policy = Eigen(
            torch.tensor(eigen_npz["policy.eigenvalues"], device=device),
            torch.tensor(eigen_npz["policy.eigenvectors"], device=device),
        )
        eigen_value_function = Eigen(
            torch.tensor(eigen_npz["value_function.eigenvalues"], device=device),
            torch.tensor(eigen_npz["value_function.eigenvectors"], device=device),
        )
        return ActorCriticEigen(eigen_policy, eigen_value_function)

    @property
    def num_eigenvectors(self) -> int:
        return min(
            self.policy.eigenvectors.shape[1], self.value_function.eigenvectors.shape[1]
        )


class HessianEigen(abc.ABC):
    def calculate_top_eigen(
        self,
        agent: stable_baselines3.common.base_class.BaseAlgorithm,
        data: Union[RolloutBufferSamples, ReplayBufferSamples],
        min_num: Union[int, Tuple[int, int]],
        eigenvectors_fullsize: bool,
    ) -> ActorCriticEigen:
        if isinstance(min_num, int):
            min_num = (min_num, min_num)
        eigen_policy = self._calculate_top_eigen_policy(agent, data, min_num[0])
        eigen_vf = self._calculate_top_eigen_vf(agent, data, min_num[1])
        if eigenvectors_fullsize:
            eigenvecs_actor, eigenvecs_critic = self._fullsize_eigenvectors(
                eigen_policy.eigenvectors, eigen_vf.eigenvectors, agent
            )
        else:
            eigenvecs_actor = eigen_policy.eigenvectors
            eigenvecs_critic = eigen_vf.eigenvectors

        return ActorCriticEigen(
            Eigen(eigen_policy.eigenvalues, eigenvecs_actor),
            Eigen(eigen_vf.eigenvalues, eigenvecs_critic),
        )

    @abc.abstractmethod
    def _calculate_top_eigen_policy(
        self,
        agent: stable_baselines3.common.base_class.BaseAlgorithm,
        data: RolloutBufferSamples,
        min_num: int,
    ) -> Eigen:
        pass

    @abc.abstractmethod
    def _calculate_top_eigen_vf(
        self,
        agent: stable_baselines3.common.base_class.BaseAlgorithm,
        data: RolloutBufferSamples,
        min_num: int,
    ) -> Eigen:
        pass

    @staticmethod
    def _fullsize_eigenvectors(
        eigenvecs_actor: torch.Tensor,
        eigenvecs_critic: torch.Tensor,
        agent: stable_baselines3.common.base_class.BaseAlgorithm,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        eigenvecs_actor_fullsize = (
            pg_subspaces.sb3_utils.common.parameters.combine_actor_critic_parameter_vectors(
                eigenvecs_actor, None, agent
            )
        )
        eigenvecs_critic_fullsize = (
            pg_subspaces.sb3_utils.common.parameters.combine_actor_critic_parameter_vectors(
                None, eigenvecs_critic, agent
            )
        )
        return eigenvecs_actor_fullsize, eigenvecs_critic_fullsize
