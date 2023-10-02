import itertools
from typing import Sequence

import stable_baselines3
import torch
from stable_baselines3.common.type_aliases import RolloutBufferSamples

from pg_subspaces.sb3_utils.common.parameters import (
    flatten_parameters,
    get_actor_critic_parameters,
    num_parameters,
)
from pg_subspaces.sb3_utils.hessian.eigen.hessian_eigen import HessianEigen, Eigen
from pg_subspaces.sb3_utils.ppo.ppo_loss import ppo_loss


class HessianEigenPowerMethod(HessianEigen):
    def __init__(self, tolerance: float, max_iter: int):
        super().__init__()
        self.tolerance = tolerance
        self.max_iter = max_iter

    def _calculate_top_eigen_policy(
        self, agent: stable_baselines3.ppo.PPO, data: RolloutBufferSamples, min_num: int
    ) -> Eigen:
        policy_params, _ = get_actor_critic_parameters(agent)
        _, loss_policy, _, _ = ppo_loss(agent, data)
        gradsH_policy = torch.autograd.grad(
            loss_policy, policy_params, create_graph=True
        )
        return self._calculate_top_eigen_net(gradsH_policy, policy_params, min_num)

    def _calculate_top_eigen_vf(
        self, agent: stable_baselines3.ppo.PPO, data: RolloutBufferSamples, min_num: int
    ) -> Eigen:
        _, vf_params = get_actor_critic_parameters(agent)
        _, _, loss_vf, _ = ppo_loss(agent, data)
        gradsH_vf = torch.autograd.grad(loss_vf, vf_params, create_graph=True)
        return self._calculate_top_eigen_net(gradsH_vf, vf_params, min_num)

    def _calculate_top_eigen_net(
        self,
        gradsH: torch.Tensor,
        params: Sequence[torch.nn.Parameter],
        min_num: int,
    ) -> Eigen:
        max_num_eigen = num_parameters(params)

        eigenvalues_pos = []
        eigenvectors_pos = []
        eigenvalues_neg = []
        eigenvectors_neg = []
        while len(eigenvalues_pos) < min_num:
            assert len(eigenvalues_pos) + len(eigenvalues_neg) <= max_num_eigen
            eigenvalue = None
            v = [
                torch.randn(p.size()).to(p.device) for p in params
            ]  # generate random vector
            v = self._normalization(v)  # normalize the vector

            for i in range(self.max_iter):
                v = self._orthonormal(
                    v, itertools.chain(eigenvectors_neg, eigenvectors_pos)
                )

                Hv = self._hessian_vector_product(gradsH, params, v)
                tmp_eigenvalue = self._group_product(Hv, v).cpu().item()

                v = self._normalization(Hv)

                if eigenvalue == None:
                    eigenvalue = tmp_eigenvalue
                else:
                    if (
                        abs(eigenvalue - tmp_eigenvalue) / (abs(eigenvalue) + 1e-6)
                        < self.tolerance
                    ):
                        break
                    else:
                        eigenvalue = tmp_eigenvalue
            # TODO: There might be better ways to extract only the positive top eigenvalues (look at scipy's eigsh)
            if eigenvalue > 0.0:
                eigenvalues_pos.append(eigenvalue)
                eigenvectors_pos.append(v)
            else:
                eigenvalues_neg.append(eigenvalue)
                eigenvectors_neg.append(v)

        return Eigen(
            torch.tensor(eigenvalues_pos),
            torch.stack([flatten_parameters(ev) for ev in eigenvectors_pos], dim=1),
        )

    @classmethod
    def _normalization(cls, v):
        """
        normalization of a list of vectors
        return: normalized vectors v
        """
        s = cls._group_product(v, v)
        s = s**0.5
        s = s.cpu().item()
        v = [vi / (s + 1e-6) for vi in v]
        return v

    @classmethod
    def _orthonormal(cls, w, v_list):
        """
        make vector w orthogonal to each vector in v_list.
        Afterwards, normalize the output w
        """
        for v in v_list:
            w = cls._group_add(w, v, alpha=-cls._group_product(w, v))
        return cls._normalization(w)

    @classmethod
    def _hessian_vector_product(cls, gradsH, params, v):
        """
        compute the hessian vector product of Hv, where
        gradsH is the gradient at the current point,
        params is the corresponding variables,
        v is the vector.
        """
        hv = torch.autograd.grad(
            gradsH, params, grad_outputs=v, only_inputs=True, retain_graph=True
        )
        return hv

    @classmethod
    def _group_product(cls, xs, ys):
        """
        the inner product of two lists of variables xs,ys
        :param xs:
        :param ys:
        :return:
        """
        return sum([torch.sum(x * y) for (x, y) in zip(xs, ys)])

    @classmethod
    def _group_add(cls, params, update, alpha=1):
        """
        params = params + update*alpha
        :param params: list of variable
        :param update: list of data
        :return:
        """
        for i, p in enumerate(params):
            params[i].data.add_(update[i] * alpha)
        return params
