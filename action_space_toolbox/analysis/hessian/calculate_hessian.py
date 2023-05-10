from typing import Callable

import stable_baselines3.common.base_class
import torch
from torch.autograd.functional import hessian
from torch.nn.utils import _stateless


def calculate_hessian(
    agent: stable_baselines3.common.base_class.BaseAlgorithm,
    loss: Callable[[stable_baselines3.common.base_class.BaseAlgorithm], torch.Tensor],
):
    def loss_reparmeterized(*params):
        names = [n for n, _ in agent.policy.named_parameters()]
        with _stateless.reparametrize_module(
            agent.policy, {n: p for n, p in zip(names, params)}
        ):
            return loss(agent)

    hess = hessian(
        loss_reparmeterized,
        tuple(agent.policy.parameters()),
    )
    return torch.cat(
        [
            torch.cat([layer.reshape(p.numel(), -1) for layer in h], dim=1)
            for h, p in zip(hess, agent.policy.parameters())
        ]
    )
