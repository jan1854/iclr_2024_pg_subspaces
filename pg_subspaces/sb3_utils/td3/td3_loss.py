from typing import Tuple

import stable_baselines3
import stable_baselines3.common.buffers
import torch
import torch.nn.functional as F
from torch.nn.utils import stateless


def td3_loss(
    agent: stable_baselines3.TD3,
    replay_data: stable_baselines3.common.buffers.ReplayBufferSamples,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        # Select action according to policy and add clipped noise
        noise = replay_data.actions.clone().data.normal_(0, agent.target_policy_noise)
        noise = noise.clamp(-agent.target_noise_clip, agent.target_noise_clip)
        next_actions = (
            agent.actor_target(replay_data.next_observations) + noise
        ).clamp(-1, 1)

        # Compute the next Q-values: min over all critics targets
        next_q_values = torch.cat(
            agent.critic_target(replay_data.next_observations, next_actions), dim=1
        )
        next_q_values, _ = torch.min(next_q_values, dim=1, keepdim=True)
        target_q_values = (
            replay_data.rewards + (1 - replay_data.dones) * agent.gamma * next_q_values
        )

    # Get current Q-values estimates for each critic network
    current_q_values = agent.critic(replay_data.observations, replay_data.actions)

    # Compute critic loss
    critic_loss = sum(
        F.mse_loss(current_q, target_q_values) for current_q in current_q_values
    )

    # This reparameterization is necessary to make sure that the actor_loss gradients do not affect the critic
    # gradients. This is not necessary in stable-baselines3 since the gradients of the critic are directly applied and
    # changes to the critic gradients afterward, thus, have no influence on the updated parameters.
    with stateless._reparametrize_module(
        agent.critic,
        {
            n: p.detach().clone().requires_grad_(True)
            for n, p in agent.critic.named_parameters()
        },
    ):
        # Compute actor loss
        actor_loss = -agent.critic.q1_forward(
            replay_data.observations, agent.actor(replay_data.observations)
        ).mean()

    return actor_loss + critic_loss, actor_loss, critic_loss
