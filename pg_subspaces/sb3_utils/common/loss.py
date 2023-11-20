from typing import List, Tuple, Union

import stable_baselines3.common.buffers
import torch
from stable_baselines3.common.type_aliases import (
    ReplayBufferSamples,
    RolloutBufferSamples,
)

from pg_subspaces.offline_rl.minimalistic_offline_sac import (
    MinimalisticOfflineSAC,
    min_offline_sac_loss,
)
from pg_subspaces.sb3_utils.common.parameters import (
    get_actor_critic_parameters,
    get_trained_parameters,
)
from pg_subspaces.sb3_utils.ppo.ppo_loss import ppo_loss
from pg_subspaces.sb3_utils.sac.sac_loss import sac_loss
from pg_subspaces.sb3_utils.td3.td3_loss import td3_loss


def actor_critic_loss(
    agent, data: Union[ReplayBufferSamples, RolloutBufferSamples]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if isinstance(agent, stable_baselines3.PPO):
        assert isinstance(data, RolloutBufferSamples)
        combined_loss, actor_loss, critic_loss, _ = ppo_loss(agent, data)
    elif isinstance(agent, stable_baselines3.TD3):
        assert isinstance(data, ReplayBufferSamples)
        combined_loss, actor_loss, critic_loss = td3_loss(agent, data)
    elif isinstance(agent, stable_baselines3.SAC):
        combined_loss, actor_loss, critic_loss = sac_loss(agent, data)
    elif isinstance(agent, MinimalisticOfflineSAC):
        combined_loss, actor_loss, critic_loss = min_offline_sac_loss(agent, data)
    else:
        raise ValueError(f"Unsupported agent {type(agent)}")
    return combined_loss, actor_loss, critic_loss


def actor_critic_gradient(
    agent: stable_baselines3.ppo.PPO,
    rollout_data: stable_baselines3.common.buffers.RolloutBufferSamples,
    all_gradients_fullsize: bool = False,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    combined_loss, policy_loss, value_function_loss = actor_critic_loss(
        agent, rollout_data
    )
    agent.policy.zero_grad()
    # TODO: This is not efficient for the gradients_fullsize=True case, combined_gradient is the sum of policy and
    #  value_function gradients (just need to be extended to the full length)
    combined_loss.backward(retain_graph=True)
    combined_gradient = [p.grad.clone() for p in get_trained_parameters(agent)]
    if all_gradients_fullsize:
        agent.policy.zero_grad()
        policy_loss.backward(retain_graph=True)
        actor_gradient = [
            p.grad.clone() if p.grad is not None else torch.zeros_like(p)
            for p in get_trained_parameters(agent)
        ]
        agent.policy.zero_grad()
        value_function_loss.backward()
        critic_gradient = [
            p.grad.clone() if p.grad is not None else torch.zeros_like(p)
            for p in get_trained_parameters(agent)
        ]
        for g_actor, g_critic in zip(actor_gradient, critic_gradient):
            assert torch.all(g_actor == 0.0) or torch.all(g_critic == 0.0)
    else:
        actor_params, critic_params = get_actor_critic_parameters(agent)
        actor_gradient = [p.grad.clone() for p in actor_params]
        critic_gradient = [p.grad.clone() for p in critic_params]
    return combined_gradient, actor_gradient, critic_gradient
