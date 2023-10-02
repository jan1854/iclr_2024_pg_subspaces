import stable_baselines3
import torch
from stable_baselines3.common.type_aliases import RolloutBufferSamples

from pg_subspaces.sb3_utils.common.loss import actor_critic_loss
from pg_subspaces.sb3_utils.hessian.calculate_hessian import calculate_hessian
from pg_subspaces.sb3_utils.hessian.eigen.hessian_eigen import Eigen, HessianEigen
from pg_subspaces.sb3_utils.ppo.ppo_parameters import (
    get_actor_parameter_names,
    get_critic_parameter_names,
)


class HessianEigenExplicit(HessianEigen):
    def _calculate_top_eigen_policy(
        self, agent: stable_baselines3.ppo.PPO, data: RolloutBufferSamples, min_num: int
    ) -> Eigen:
        names_policy = get_actor_parameter_names(agent.policy.named_parameters())
        hess_policy = calculate_hessian(
            agent, lambda a: actor_critic_loss(a, data)[1], names_policy
        )
        with torch.no_grad():
            eigenvalues_policy, eigenvectors_policy = torch.linalg.eigh(hess_policy)
            eigenvalues_policy = torch.flip(eigenvalues_policy, dims=(0,))
            eigenvectors_policy = torch.flip(eigenvectors_policy, dims=(1,))
        return Eigen(eigenvalues_policy, eigenvectors_policy)

    def _calculate_top_eigen_vf(
        self, agent: stable_baselines3.ppo.PPO, data: RolloutBufferSamples, min_num: int
    ) -> Eigen:
        names_vf = get_critic_parameter_names(agent.policy.named_parameters())
        hess_vf = calculate_hessian(
            agent, lambda a: actor_critic_loss(agent, data)[2], names_vf
        )
        with torch.no_grad():
            eigenvalues_vf, eigenvectors_vf = torch.linalg.eigh(hess_vf)
            eigenvalues_vf = torch.flip(eigenvalues_vf, dims=(0,))
            eigenvectors_vf = torch.flip(eigenvectors_vf, dims=(1,))
        return Eigen(eigenvalues_vf, eigenvectors_vf)
