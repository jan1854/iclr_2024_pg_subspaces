from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import stable_baselines3.common.base_class
import torch


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
    if isinstance(agent, stable_baselines3.PPO):
        return list(agent.policy.parameters())
    elif isinstance(agent, stable_baselines3.TD3) or isinstance(
        agent, stable_baselines3.SAC
    ):
        return list(p for n, p in agent.policy.named_parameters() if "_target" not in n)
    else:
        raise ValueError(f"Unsupported agent of type {type(agent)}")


def unflatten_parameters_for_agent(
    param_vec: torch.Tensor,
    agent: stable_baselines3.common.base_class.BaseAlgorithm,
) -> List[torch.Tensor]:
    return unflatten_parameters(param_vec, [p.shape for p in agent.policy.parameters()])


def get_actor_critic_parameters(
    agent: stable_baselines3.common.base_class.BaseAlgorithm,
) -> Tuple[List[torch.nn.Parameter], List[torch.nn.Parameter]]:
    if isinstance(agent, stable_baselines3.PPO):
        actor_parameter_names = pg_subspaces.sb3_utils.ppo.ppo_parameters.get_actor_parameter_names(
            agent.policy.named_parameters()
        )
        critic_parameter_names = (
            pg_subspaces.sb3_utils.ppo.ppo_parameters.get_critic_parameter_names(
                agent.policy.named_parameters()
            )
        )
        actor_parameters = [
            p for n, p in agent.policy.named_parameters() if n in actor_parameter_names
        ]
        critic_parameters = [
            p for n, p in agent.policy.named_parameters() if n in critic_parameter_names
        ]
        return actor_parameters, critic_parameters
    elif isinstance(agent, stable_baselines3.TD3) or isinstance(
        agent, stable_baselines3.SAC
    ):
        return list(agent.policy.actor.parameters()), list(
            agent.policy.critic.parameters()
        )
    else:
        raise ValueError(f"Unsupported agent: {type(agent)}")


def combine_actor_critic_parameter_vectors(
    policy_parameters: Optional[torch.Tensor],
    value_function_parameters: Optional[torch.Tensor],
    agent: stable_baselines3.common.base_class.BaseAlgorithm,
) -> torch.Tensor:
    num_parameters_agent = sum([p.numel() for p in get_trained_parameters(agent)])
    assert policy_parameters is None or len(policy_parameters) < num_parameters_agent
    assert (
        value_function_parameters is None
        or len(value_function_parameters) < num_parameters_agent
    )
    assert (
        policy_parameters is None
        or value_function_parameters is None
        or len(policy_parameters) + len(value_function_parameters)
        == num_parameters_agent
    )
    policy_idx = 0
    vf_idx = 0
    device = (
        policy_parameters.device
        if policy_parameters is not None
        else value_function_parameters.device
    )
    shape = (
        policy_parameters.shape[1:]
        if policy_parameters is not None
        else value_function_parameters.shape[1:]
    )
    parameters = []
    for name, params in agent.policy.named_parameters():
        if "target" in name:
            continue
        elif (
            "action_net" in name
            or "policy_net" in name
            or name == "log_std"
            or name.startswith("actor")
        ):
            if policy_parameters is not None:
                parameters.append(
                    policy_parameters[policy_idx : policy_idx + params.numel()]
                )
                policy_idx += params.numel()
            else:
                parameters.append(torch.zeros((params.numel(),) + shape, device=device))
        elif "value_net" in name or name.startswith("critic"):
            if value_function_parameters is not None:
                parameters.append(
                    value_function_parameters[vf_idx : vf_idx + params.numel()]
                )
                vf_idx += params.numel()
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
