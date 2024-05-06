from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import stable_baselines3.common.base_class
import torch

from pg_subspaces.offline_rl.minimalistic_offline_sac import MinimalisticOfflineSAC
from pg_subspaces.sb3_utils.ppo.ppo_parameters import (
    get_actor_parameter_names,
    get_critic_parameter_names,
)


def flatten_parameters(param_seq: Iterable[torch.nn.Parameter]) -> torch.Tensor:
    return torch.cat([s.flatten() for s in param_seq])


def num_parameters(param_seq: Iterable[torch.nn.Parameter]) -> int:
    return sum(p.numel() for p in param_seq)


def unflatten_parameters(param_vec: torch.Tensor, shapes: Sequence[Sequence[int]]):
    param_seq = []
    vec_idx = 0
    for shape in shapes:
        param_seq.append(param_vec[vec_idx : vec_idx + np.prod(shape)].reshape(shape))
        vec_idx += np.prod(shape)
    return param_seq


def get_trained_parameters(
    agent: stable_baselines3.common.base_class.BaseAlgorithm,
) -> List[torch.nn.Parameter]:
    return [p for n, p in get_trained_named_parameters(agent)]


def get_trained_named_parameters(
    agent: stable_baselines3.common.base_class.BaseAlgorithm,
) -> List[Tuple[str, torch.nn.Parameter]]:
    if isinstance(agent, stable_baselines3.PPO):
        return list(agent.policy.named_parameters())
    elif isinstance(agent, stable_baselines3.TD3) or isinstance(
        agent, stable_baselines3.SAC
    ):
        return [
            (n, p) for n, p in agent.policy.named_parameters() if "_target" not in n
        ]
    else:
        raise ValueError(f"Unsupported agent of type {type(agent)}")


def unflatten_parameters_for_agent(
    param_vec: torch.Tensor,
    agent: stable_baselines3.common.base_class.BaseAlgorithm,
) -> List[torch.Tensor]:
    return unflatten_parameters(param_vec, [p.shape for p in agent.policy.parameters()])


def get_actor_critic_parameter_names(
    agent: stable_baselines3.common.base_class.BaseAlgorithm,
) -> Tuple[List[str], List[str]]:
    if isinstance(agent, stable_baselines3.PPO):
        parameter_names = [n for n, _ in agent.policy.named_parameters()]
        actor_parameter_names = get_actor_parameter_names(parameter_names)
        critic_parameter_names = get_critic_parameter_names(parameter_names)
        actor_parameters_names = [
            n for n, _ in agent.policy.named_parameters() if n in actor_parameter_names
        ]
        critic_parameters_names = [
            n for n, _ in agent.policy.named_parameters() if n in critic_parameter_names
        ]
        return actor_parameters_names, critic_parameters_names
    elif (
        isinstance(agent, stable_baselines3.TD3)
        or isinstance(agent, stable_baselines3.SAC)
        or isinstance(agent, MinimalisticOfflineSAC)
    ):
        return ["actor." + n for n, _ in agent.policy.actor.named_parameters()], list(
            ["critic." + n for n, _ in agent.policy.critic.named_parameters()]
        )
    else:
        raise ValueError(f"Unsupported agent: {type(agent)}")


def get_actor_critic_parameters(
    agent: stable_baselines3.common.base_class.BaseAlgorithm,
) -> Tuple[List[torch.nn.Parameter], List[torch.nn.Parameter]]:
    actor_parameter_names, critic_parameter_names = get_actor_critic_parameter_names(
        agent
    )
    actor_parameters = [
        p for n, p in agent.policy.named_parameters() if n in actor_parameter_names
    ]
    critic_parameters = [
        p for n, p in agent.policy.named_parameters() if n in critic_parameter_names
    ]
    return actor_parameters, critic_parameters


def combine_actor_critic_parameters(
    actor_parameters: Optional[Sequence[torch.Tensor]],
    critic_parameters: Optional[Sequence[torch.Tensor]],
    agent: stable_baselines3.common.base_class.BaseAlgorithm,
) -> List[torch.Tensor]:
    actor_idx = 0
    critic_idx = 0
    actor_parameter_names, critic_parameter_names = get_actor_critic_parameter_names(
        agent
    )
    parameters = []
    for name, params in agent.policy.named_parameters():
        if "target" in name:
            continue
        elif name in actor_parameter_names:
            if actor_parameters is not None:
                assert actor_parameters[actor_idx].shape == params.shape
                parameters.append(actor_parameters[actor_idx])
                actor_idx += 1
            else:
                parameters.append(torch.zeros_like(params))
        elif name in critic_parameter_names:
            if critic_parameters is not None:
                assert critic_parameters[critic_idx].shape == params.shape
                parameters.append(critic_parameters[critic_idx])
                critic_idx += 1
            else:
                parameters.append(torch.zeros_like(params))
        else:
            raise ValueError(f"Encountered invalid parameter: {name}")
    assert actor_parameters is None or actor_idx == len(actor_parameters)
    assert critic_parameters is None or critic_idx == len(critic_parameters)
    return parameters


def combine_actor_critic_parameter_vectors(
    actor_parameters: Optional[torch.Tensor],
    critic_parameters: Optional[torch.Tensor],
    agent: stable_baselines3.common.base_class.BaseAlgorithm,
) -> torch.Tensor:
    num_parameters_agent = sum([p.numel() for p in get_trained_parameters(agent)])
    assert actor_parameters is None or len(actor_parameters) < num_parameters_agent
    assert critic_parameters is None or len(critic_parameters) < num_parameters_agent
    assert (
        actor_parameters is None
        or critic_parameters is None
        or len(actor_parameters) + len(critic_parameters) == num_parameters_agent
    )
    actor_idx = 0
    critic_idx = 0
    device = (
        actor_parameters.device
        if actor_parameters is not None
        else critic_parameters.device
    )
    shape = (
        actor_parameters.shape[1:]
        if actor_parameters is not None
        else critic_parameters.shape[1:]
    )
    parameters = []
    for name, params in agent.policy.named_parameters():
        if "target" in name:
            continue
        # TODO: Switch to using get_actor_critic_parameter_names here
        elif (
            "action_net" in name
            or "policy_net" in name
            or name == "log_std"
            or name.startswith("actor")
        ):
            if actor_parameters is not None:
                parameters.append(
                    actor_parameters[actor_idx : actor_idx + params.numel()]
                )
                actor_idx += params.numel()
            else:
                parameters.append(torch.zeros((params.numel(),) + shape, device=device))
        elif "value_net" in name or name.startswith("critic"):
            if critic_parameters is not None:
                parameters.append(
                    critic_parameters[critic_idx : critic_idx + params.numel()]
                )
                critic_idx += params.numel()
            else:
                parameters.append(torch.zeros((params.numel(),) + shape, device=device))
        else:
            raise ValueError(f"Encountered invalid parameter: {name}")
    return torch.cat(parameters, dim=0)


def project(
    vec: torch.Tensor, subspace: torch.Tensor, result_in_orig_space: bool
) -> torch.Tensor:
    # Projection matrix: (subspace^T @ subspace)^(-1) @ subspace^T
    vec_subspace = torch.linalg.solve(subspace.T @ subspace, subspace.T @ vec).to(
        torch.float32
    )
    if result_in_orig_space:
        return subspace @ vec_subspace
    else:
        return vec_subspace


def project_orthonormal(
    vec: torch.Tensor, subspace: torch.Tensor, result_in_orig_space: bool
) -> torch.Tensor:
    vec_subspace = subspace.T @ vec
    if result_in_orig_space:
        return subspace @ vec_subspace
    else:
        return vec_subspace


def project_orthonormal_inverse(
    vec_subspace: torch.Tensor,
    subspace: torch.Tensor,
) -> torch.Tensor:
    return subspace @ vec_subspace
