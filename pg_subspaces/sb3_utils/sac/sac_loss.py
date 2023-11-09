from typing import Tuple

import stable_baselines3.common.buffers
import torch
from torch.nn import functional as F


def sac_loss(
    agent: stable_baselines3.SAC,
    replay_data: stable_baselines3.common.buffers.ReplayBufferSamples,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    actions_pi, log_prob = agent.actor.action_log_prob(replay_data.observations)
    log_prob = log_prob.reshape(-1, 1)

    if agent.ent_coef_optimizer is not None:
        # Important: detach the variable from the graph
        # so we don't change it with other losses
        # see https://github.com/rail-berkeley/softlearning/issues/60
        ent_coef = torch.exp(agent.log_ent_coef.detach())
    else:
        ent_coef = agent.ent_coef_tensor

    with torch.no_grad():
        # Select action according to policy
        next_actions, next_log_prob = agent.actor.action_log_prob(
            replay_data.next_observations
        )
        # Compute the next Q values: min over all critics targets
        next_q_values = torch.cat(
            agent.critic_target(replay_data.next_observations, next_actions), dim=1
        )
        next_q_values, _ = torch.min(next_q_values, dim=1, keepdim=True)
        # add entropy term
        next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
        # td error + entropy term
        target_q_values = (
            replay_data.rewards + (1 - replay_data.dones) * agent.gamma * next_q_values
        )

    # Get current Q-values estimates for each critic network
    # using action from the replay buffer
    current_q_values = agent.critic(replay_data.observations, replay_data.actions)

    # Compute critic loss
    critic_loss = 0.5 * sum(
        F.mse_loss(current_q, target_q_values) for current_q in current_q_values
    )

    # Compute actor loss
    # Alternative: actor_loss = th.mean(log_prob - qf1_pi)
    # Min over all critic networks
    with torch.no_grad():
        q_values_pi = torch.cat(
            agent.critic(replay_data.observations, actions_pi), dim=1
        )
        min_qf_pi, _ = torch.min(q_values_pi, dim=1, keepdim=True)
    actor_loss = (ent_coef * log_prob - min_qf_pi).mean()

    return actor_loss + critic_loss, actor_loss, critic_loss
