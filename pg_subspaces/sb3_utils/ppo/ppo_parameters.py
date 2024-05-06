from typing import List, Optional, Sequence, Tuple

import stable_baselines3
import torch


def combine_actor_critic_parameter_vectors(
    policy_parameters: Optional[torch.Tensor],
    value_function_parameters: Optional[torch.Tensor],
    agent: stable_baselines3.ppo.PPO,
) -> torch.Tensor:
    num_parameters_agent = sum([p.numel() for p in agent.policy.parameters()])
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
        if "action_net" in name or "policy_net" in name or name == "log_std":
            if policy_parameters is not None:
                parameters.append(
                    policy_parameters[policy_idx : policy_idx + params.numel()]
                )
                policy_idx += params.numel()
            else:
                parameters.append(torch.zeros((params.numel(),) + shape, device=device))
        elif "value_net" in name:
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


def get_actor_parameter_names(parameter_names: Sequence[str]) -> List[str]:
    return [
        n
        for n in parameter_names
        if "action_net" in n or "policy_net" in n or n == "log_std"
    ]


def get_critic_parameter_names(parameter_names: Sequence[str]) -> List[str]:
    return [n for n in parameter_names if "value_net" in n]
