import logging
from typing import Optional, Sequence

import numpy as np
import scipy
import stable_baselines3.common.base_class
import torch
import scipy.sparse.linalg

from pg_subspaces.sb3_utils.common.loss import actor_critic_loss
from pg_subspaces.sb3_utils.common.parameters import (
    get_actor_critic_parameters,
    num_parameters,
    flatten_parameters,
    unflatten_parameters,
)
from pg_subspaces.sb3_utils.hessian.eigen.hessian_eigen import Eigen, HessianEigen
from stable_baselines3.common.buffers import RolloutBufferSamples

logger = logging.getLogger(__name__)


class HessianEigenLanczos(HessianEigen):
    def __init__(
        self,
        tolerance: float,
        max_iter: int,
        num_lanczos_vectors: Optional[int],
        which_eigen: str = "LA",
    ):
        super().__init__()
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.num_lanczos_vectors = num_lanczos_vectors
        self.which_eigen = which_eigen

    def _calculate_top_eigen_policy(
        self,
        agent: stable_baselines3.common.base_class.BaseAlgorithm,
        data: RolloutBufferSamples,
        min_num: int,
    ) -> Eigen:
        actor_parameters, _ = get_actor_critic_parameters(agent)
        _, actor_loss, _ = actor_critic_loss(agent, data)
        return self._calculate_top_eigen_net(actor_loss, actor_parameters, min_num)

    def _calculate_top_eigen_vf(
        self,
        agent: stable_baselines3.common.base_class.BaseAlgorithm,
        data: RolloutBufferSamples,
        min_num: int,
    ) -> Eigen:
        _, critic_parameters = get_actor_critic_parameters(agent)
        _, _, critic_loss = actor_critic_loss(agent, data)
        return self._calculate_top_eigen_net(critic_loss, critic_parameters, min_num)

    def _calculate_top_eigen_net(
        self,
        loss: torch.Tensor,
        params: Sequence[torch.nn.Parameter],
        min_num: int,
    ) -> Eigen:
        gradient = torch.autograd.grad(loss, params, create_graph=True)

        if (
            self.num_lanczos_vectors is not None
            and self.num_lanczos_vectors < 2 * min_num
        ):
            logger.warning(
                f"Number of Lanczos vectors should usually be > 2 * min_num, but got min_num: {min_num}, "
                f"num_lanczos_vectors: {self.num_lanczos_vectors}"
            )

        def hessian_vector_product(vec: np.ndarray) -> np.ndarray:
            vec = torch.from_numpy(vec).to(loss.device)
            vec = unflatten_parameters(vec, [p.shape for p in params])
            hessian_vector_prod = torch.autograd.grad(
                gradient, params, grad_outputs=vec, retain_graph=True
            )
            return flatten_parameters(hessian_vector_prod).cpu().numpy()

        num_params = num_parameters(params)
        hess_vec_prod_operator = scipy.sparse.linalg.LinearOperator(
            (num_params, num_params), hessian_vector_product
        )

        eigenvals, eigenvecs = scipy.sparse.linalg.eigsh(
            A=hess_vec_prod_operator,
            k=min_num,
            which=self.which_eigen,
            maxiter=self.max_iter,
            tol=self.tolerance,
            ncv=self.num_lanczos_vectors,
            return_eigenvectors=True,
        )
        # The results of eigsh are returned in ascending order (irrespective of the fact that we calculate the largest
        # eigenvalues). Flip the results to have the largest eigenvalues first.
        eigenvals = np.flip(eigenvals)
        eigenvecs = np.flip(eigenvecs, axis=1)
        # Copy is needed since torch.from_numpy cannot deal with negative strides
        return Eigen(
            torch.tensor(eigenvals.copy()).to(loss.device),
            torch.from_numpy(eigenvecs.copy()).to(loss.device),
        )
