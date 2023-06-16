from typing import Callable, Sequence, Optional

import stable_baselines3.common.base_class
import torch
from torch.autograd.functional import hessian
from torch.nn.utils import _stateless


def calculate_hessian(
    agent: stable_baselines3.common.base_class.BaseAlgorithm,
    loss: Callable[[stable_baselines3.common.base_class.BaseAlgorithm], torch.Tensor],
    parameter_names: Optional[Sequence[str]] = None,
):
    if parameter_names is None:
        parameter_names = [n for n, _ in agent.policy.named_parameters()]
    parameters = [p for n, p in agent.policy.named_parameters() if n in parameter_names]

    def loss_reparametrized(*params):
        with _stateless.reparametrize_module(
            agent.policy, {n: p for n, p in zip(parameter_names, params)}
        ):
            return loss(agent)

    hess = hessian(
        loss_reparametrized,
        tuple(parameters),
    )
    return torch.cat(
        [
            torch.cat([layer.reshape(p.numel(), -1) for layer in h], dim=1)
            for h, p in zip(hess, parameters)
        ]
    )
