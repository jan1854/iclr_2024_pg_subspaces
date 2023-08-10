import re
from pathlib import Path
from typing import Literal, Optional, Sequence, Tuple, Union

import stable_baselines3
import torch
from sb3_utils.hessian.eigen.hessian_eigen import ActorCriticEigen, HessianEigen
from sb3_utils.common.parameters import flatten_parameters
from stable_baselines3.common.type_aliases import RolloutBufferSamples

from sb3_utils.ppo.ppo_parameters import combine_actor_critic_parameter_vectors


class CachedEigenIterator:
    def __init__(
        self,
        hessian_calc: "HessianEigenCachedCalculator",
        agent: stable_baselines3.ppo.PPO,
        loss_name: Literal["policy_loss", "value_function_loss", "combined_loss"],
        env_steps: Sequence[int],
        num_grad_steps_additional_training: int,
    ):
        self.hessian_calc = hessian_calc
        self.agent = agent
        self.loss_name = loss_name
        self.env_steps = tuple(env_steps)
        self.num_grad_steps_additional_training = num_grad_steps_additional_training
        self._idx = 0

    def __iter__(self) -> "CachedEigenIterator":
        return self

    def __next__(self) -> Tuple[int, torch.Tensor, torch.Tensor]:
        if self._idx < len(self.env_steps):
            env_step = self.env_steps[self._idx]
            eigen = self.hessian_calc.read_cached_eigen(
                env_step, self.num_grad_steps_additional_training
            )
            if self.loss_name == "combined_loss":
                eigenvals, eigenvecs = self.hessian_calc.collect_top_eigenvectors(
                    self.agent, eigen
                )
            elif self.loss_name == "policy_loss":
                eigenvals = eigen.policy.eigenvalues
                eigenvecs = combine_actor_critic_parameter_vectors(
                    eigen.policy.eigenvectors, None, self.agent
                )
            elif self.loss_name == "value_function_loss":
                eigenvals = eigen.value_function.eigenvalues
                eigenvecs = combine_actor_critic_parameter_vectors(
                    None, eigen.value_function.eigenvectors, self.agent
                )
            else:
                raise ValueError(f"Unknown loss: {self.loss_name}.")
            self._idx += 1
            return env_step, eigenvals, eigenvecs
        else:
            raise StopIteration


class HessianEigenCachedCalculator:
    def __init__(
        self,
        run_dir: Path,
        hessian_eigen: HessianEigen,
        num_eigenvectors_to_cache: int = 200,
        device: Union[str, torch.device] = "cpu",
    ):
        self.cache_path = run_dir / "cached_results" / "eigen"
        self.cache_path.mkdir(exist_ok=True, parents=True)
        self.hessian_eigen = hessian_eigen
        self.num_eigenvectors_to_cache = num_eigenvectors_to_cache
        self.device = device

    def _get_eigen(
        self,
        agent: stable_baselines3.ppo.PPO,
        data: RolloutBufferSamples,
        env_step: int,
        num_grad_steps_additional_training: int,
        num_eigenvectors: Optional[int],
        overwrite_cache: bool,
        calculate_if_no_cached_value: bool,
    ) -> ActorCriticEigen:
        # Check that the there is no parameter sharing between policy and value function
        assert (
            sum(["shared" in name for name, _ in agent.policy.named_parameters()]) == 0
        )
        cached_eigen = self.read_cached_eigen(
            env_step, num_grad_steps_additional_training
        )
        # TODO: This does not check if the cache contains all eigenvectors (might be smaller than num_eigenvectors)
        if (
            not overwrite_cache
            and cached_eigen is not None
            and (
                num_eigenvectors is None
                or cached_eigen.num_eigenvectors >= num_eigenvectors
            )
        ):
            eigen = cached_eigen
        else:
            if not calculate_if_no_cached_value:
                raise FileNotFoundError(f"Did not find cache file.")

            eigen = self.hessian_eigen.calculate_top_eigen(
                agent, data, self.num_eigenvectors_to_cache, eigenvectors_fullsize=False
            )
            self.cache_eigen(eigen, env_step, num_grad_steps_additional_training)
        eigen.policy.eigenvectors = eigen.policy.eigenvectors[:, :num_eigenvectors]
        eigen.value_function.eigenvectors = eigen.value_function.eigenvectors[
            :, :num_eigenvectors
        ]
        return eigen

    def get_eigen_combined_loss(
        self,
        agent: stable_baselines3.ppo.PPO,
        data: RolloutBufferSamples,
        env_step: int,
        num_grad_steps_additional_training: int = 0,
        num_eigenvectors: Optional[int] = None,
        overwrite_cache: bool = False,
        calculate_if_no_cached_value: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        eigen = self._get_eigen(
            agent,
            data,
            env_step,
            num_grad_steps_additional_training,
            num_eigenvectors,
            overwrite_cache,
            calculate_if_no_cached_value,
        )
        return self.collect_top_eigenvectors(agent, eigen, num_eigenvectors)

    def get_eigen_policy_vf_loss(
        self,
        agent: stable_baselines3.ppo.PPO,
        data: RolloutBufferSamples,
        env_step: int,
        num_grad_steps_additional_training: int = 0,
        num_eigenvectors: Optional[int] = None,
        overwrite_cache: bool = False,
        calculate_if_no_cached_value: bool = True,
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        eigen = self._get_eigen(
            agent,
            data,
            env_step,
            num_grad_steps_additional_training,
            num_eigenvectors,
            overwrite_cache,
            calculate_if_no_cached_value,
        )
        policy_eigenvectors_all_parameters = combine_actor_critic_parameter_vectors(
            eigen.policy.eigenvectors, None, agent
        )
        vf_eigenvectors_all_parameters = combine_actor_critic_parameter_vectors(
            None, eigen.value_function.eigenvectors, agent
        )
        return (eigen.policy.eigenvalues, policy_eigenvectors_all_parameters), (
            eigen.value_function.eigenvalues,
            vf_eigenvectors_all_parameters,
        )

    @classmethod
    def collect_top_eigenvectors(
        cls,
        agent: stable_baselines3.ppo.PPO,
        eigen: ActorCriticEigen,
        num_eigenvectors: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        eigenvectors = []
        policy_ev_idx = 0
        vf_ev_idx = 0
        if num_eigenvectors is None:
            num_eigenvectors = eigen.num_eigenvectors

        # Make sure that eigen either contains the required number of eigenvectors from both the policy and value
        # function or all eigenvectors (otherwise we cannot ensure that the eigenvectors that we collected are the top
        # eigenvectors of the combined loss).
        assert (
            num_eigenvectors <= eigen.num_eigenvectors
            or eigen.policy.eigenvectors.shape[1]
            + eigen.value_function.eigenvectors.shape[1]
            == len(flatten_parameters(agent.policy.parameters()))
        )

        # Select the eigenvectors with the largest eigenvalues from the eigenvectors for the policy and value function.
        for _ in range(num_eigenvectors):
            if (
                policy_ev_idx < len(eigen.policy.eigenvalues)
                and eigen.policy.eigenvalues[policy_ev_idx]
                > eigen.value_function.eigenvalues[vf_ev_idx]
            ):
                eigenvectors.append(
                    combine_actor_critic_parameter_vectors(
                        eigen.policy.eigenvectors[:, policy_ev_idx], None, agent
                    )
                )
                policy_ev_idx += 1
            elif vf_ev_idx < len(eigen.value_function.eigenvalues):
                eigenvectors.append(
                    combine_actor_critic_parameter_vectors(
                        None, eigen.value_function.eigenvectors[:, vf_ev_idx], agent
                    )
                )
                vf_ev_idx += 1
            else:
                break
        eigenvectors = torch.stack(eigenvectors, dim=1)
        # TODO: Not efficient since the individual tensors are already sorted
        eigenvalues = (
            torch.cat((eigen.policy.eigenvalues, eigen.value_function.eigenvalues))
            .sort(descending=True)
            .values
        )
        return eigenvalues, eigenvectors

    def read_cached_eigen(
        self, env_step: int, num_grad_steps_additional_training: int
    ) -> Optional[ActorCriticEigen]:
        cache_path = self._get_cache_path(
            self.cache_path, env_step, num_grad_steps_additional_training
        )
        return self.read_eigen_from_path(cache_path, self.device)

    @classmethod
    def read_eigen_from_path(
        cls, cache_path: Path, device: torch.device
    ) -> Optional[ActorCriticEigen]:
        if not cache_path.exists():
            return None
        else:
            return ActorCriticEigen.load(cache_path, device)

    def cache_eigen(
        self,
        eigen: ActorCriticEigen,
        env_step: int,
        num_grad_steps_additional_training: int,
    ) -> None:
        cache_path = self._get_cache_path(
            self.cache_path, env_step, num_grad_steps_additional_training
        )
        eigen.dump(cache_path, self.num_eigenvectors_to_cache)

    def iter_cached_eigen(
        self,
        agent: stable_baselines3.ppo.PPO,
        num_grad_steps_additional_training: int = 0,
        loss_name: Literal[
            "policy_loss", "value_function_loss", "combined_loss"
        ] = "combined_loss",
    ) -> "CachedEigenIterator":
        if num_grad_steps_additional_training == 0:
            pattern = "eigen_[0-9]+.npz"
        else:
            pattern = f"eigen_[0-9]+_additional_grad_steps_{num_grad_steps_additional_training:05d}.npz"
        env_steps = [
            # TODO: Does not work if the number of env_steps is > 10000000 --> Use re again
            int(cache_file.name[len("eigen") + 1 : len("eigen") + 8])
            for cache_file in self.cache_path.iterdir()
            if re.fullmatch(pattern, cache_file.name)
        ]
        return CachedEigenIterator(
            self,
            agent,
            loss_name,
            sorted(env_steps),
            num_grad_steps_additional_training,
        )

    @classmethod
    def _get_cache_path(
        cls, cache_path: Path, env_step: int, num_grad_steps_additional_training: int
    ) -> Path:
        path = cache_path / f"eigen_{env_step:07d}"
        if num_grad_steps_additional_training > 0:
            path = path.with_name(
                path.name
                + f"_additional_grad_steps_{num_grad_steps_additional_training:05d}"
            )
        return path.with_suffix(".npz")
