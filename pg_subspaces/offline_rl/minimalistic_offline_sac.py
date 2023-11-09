import io
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type, Union

import numpy as np
import torch
import torch.nn.functional as F
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import (
    Schedule,
    ReplayBufferSamples,
    MaybeCallback,
)
from stable_baselines3.common.utils import get_parameters_by_name, polyak_update
from stable_baselines3.sac.policies import (
    SACPolicy,
    CnnPolicy,
    MlpPolicy,
    MultiInputPolicy,
)

from pg_subspaces.offline_rl.offline_algorithm import OfflineAlgorithm


class MinimalisticOfflineSAC(OfflineAlgorithm):
    policy_aliases: Dict[str, Type[BasePolicy]] = {
        "MlpPolicy": MlpPolicy,
        "CnnPolicy": CnnPolicy,
        "MultiInputPolicy": MultiInputPolicy,
    }

    def __init__(
        self,
        policy: Union[str, Type[SACPolicy]],
        dataset: ReplayBuffer,
        learning_rate: Union[float, Schedule] = 3e-4,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        actor_rl_objective_weight: float = 2.5,
        action_noise: Optional[ActionNoise] = None,
        ent_coef: Union[str, float] = "auto",
        target_update_interval: int = 1,
        target_entropy: Union[str, float] = "auto",
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        stats_window_size: int = 100,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[torch.device, str] = "auto",
        _init_setup_model: bool = True,
    ):
        super().__init__(
            policy,
            dataset,
            learning_rate,
            batch_size,
            tau,
            gamma,
            action_noise,
            tensorboard_log,
            policy_kwargs,
            stats_window_size,
            verbose,
            seed,
            device,
        )
        self.actor_rl_objective_weight = actor_rl_objective_weight
        self.ent_coef = ent_coef
        self.target_update_interval = target_update_interval
        self.target_entropy = target_entropy
        self.tensorboard_log = tensorboard_log

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super()._setup_model()
        self._create_aliases()
        # Running mean and running var
        self.batch_norm_stats = get_parameters_by_name(self.critic, ["running_"])
        self.batch_norm_stats_target = get_parameters_by_name(
            self.critic_target, ["running_"]
        )
        # Target entropy is used when learning the entropy coefficient
        if self.target_entropy == "auto":
            # automatically set target entropy if needed
            self.target_entropy = -np.prod(self.action_space.shape).astype(np.float32)
        else:
            # Force conversion
            # this will also throw an error for unexpected string
            self.target_entropy = float(self.target_entropy)

        # The entropy coefficient or entropy can be learned automatically
        # see Automating Entropy Adjustment for Maximum Entropy RL section
        # of https://arxiv.org/abs/1812.05905
        if isinstance(self.ent_coef, str) and self.ent_coef.startswith("auto"):
            # Default initial value of ent_coef when learned
            init_value = 1.0
            if "_" in self.ent_coef:
                init_value = float(self.ent_coef.split("_")[1])
                assert (
                    init_value > 0.0
                ), "The initial value of ent_coef must be greater than 0"

            # Note: we optimize the log of the entropy coeff which is slightly different from the paper
            # as discussed in https://github.com/rail-berkeley/softlearning/issues/37
            self.log_ent_coef = torch.log(
                torch.ones(1, device=self.device) * init_value
            ).requires_grad_(True)
            self.ent_coef_optimizer = torch.optim.Adam(
                [self.log_ent_coef], lr=self.lr_schedule(1)
            )
        else:
            # Force conversion to float
            # this will throw an error if a malformed string (different from 'auto')
            # is passed
            self.ent_coef_tensor = torch.tensor(float(self.ent_coef)).to(self.device)

    def _create_aliases(self) -> None:
        self.actor = self.policy.actor
        self.critic = self.policy.critic
        self.critic_target = self.policy.critic_target

    def train(self, batch: ReplayBufferSamples) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizers learning rate
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]

        # Update learning rate according to lr schedule
        self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []

        # Action by the current actor for the sampled state
        actions_pi, log_prob = self.actor.action_log_prob(batch.observations)
        log_prob = log_prob.reshape(-1, 1)

        ent_coef_loss = None
        if self.ent_coef_optimizer is not None:
            # Important: detach the variable from the graph
            # so we don't change it with other losses
            # see https://github.com/rail-berkeley/softlearning/issues/60
            ent_coef = torch.exp(self.log_ent_coef.detach())
            ent_coef_loss = -(
                self.log_ent_coef * (log_prob + self.target_entropy).detach()
            ).mean()
            ent_coef_losses.append(ent_coef_loss.item())
        else:
            ent_coef = self.ent_coef_tensor

        ent_coefs.append(ent_coef.item())

        # Optimize entropy coefficient, also called
        # entropy temperature or alpha in the paper
        if ent_coef_loss is not None:
            self.ent_coef_optimizer.zero_grad()
            ent_coef_loss.backward()
            self.ent_coef_optimizer.step()

        with torch.no_grad():
            # Select action according to policy
            next_actions, next_log_prob = self.actor.action_log_prob(
                batch.next_observations
            )
            # Compute the next Q values: min over all critics targets
            next_q_values = torch.cat(
                self.critic_target(batch.next_observations, next_actions),
                dim=1,
            )
            next_q_values, _ = torch.min(next_q_values, dim=1, keepdim=True)
            # add entropy term
            next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
            # td error + entropy term
            target_q_values = (
                batch.rewards + (1 - batch.dones) * self.gamma * next_q_values
            )

        # Get current Q-values estimates for each critic network
        # using action from the replay buffer
        current_q_values = self.critic(batch.observations, batch.actions)

        # Compute critic loss
        critic_loss = 0.5 * sum(
            F.mse_loss(current_q, target_q_values) for current_q in current_q_values
        )
        critic_losses.append(critic_loss.item())

        # Optimize the critic
        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        self.critic.optimizer.step()

        # Compute actor loss
        # Alternative: actor_loss = torch.mean(log_prob - qf1_pi)
        # Min over all critic networks
        q_values_pi = torch.cat(self.critic(batch.observations, actions_pi), dim=1)
        min_qf_pi, _ = torch.min(q_values_pi, dim=1, keepdim=True)
        policy_mean, policy_log_std, _ = self.actor.get_action_dist_params(
            batch.observations
        )
        self.actor.action_dist.proba_distribution(policy_mean, policy_log_std)
        bc_log_prob = self.actor.action_dist.log_prob(batch.actions)
        policy_entropy_loss = (ent_coef * log_prob).mean()
        rl_objective_weight_scaled = (
            self.actor_rl_objective_weight / min_qf_pi.detach().abs().mean()
        )
        policy_q_value_loss = -(rl_objective_weight_scaled * min_qf_pi).mean()
        policy_bc_loss = -bc_log_prob.mean()
        actor_loss = policy_entropy_loss + policy_q_value_loss + policy_bc_loss
        actor_losses.append(actor_loss.item())

        # Optimize the actor
        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        self.actor.optimizer.step()

        # Update target networks
        if self._n_updates % self.target_update_interval == 0:
            polyak_update(
                self.critic.parameters(), self.critic_target.parameters(), self.tau
            )
            # Copy running stats, see GH issue #996
            polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

        self._n_updates += 1

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))
        self.logger.record("train/actor_loss_entropy", policy_entropy_loss.item())
        self.logger.record("train/actor_loss_q_value", policy_q_value_loss.item())
        self.logger.record("train/actor_loss_behavioral_cloning", policy_bc_loss.item())

    def _excluded_save_params(self) -> List[str]:
        return super()._excluded_save_params() + ["actor", "critic", "critic_target"]

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy", "actor.optimizer", "critic.optimizer"]
        if self.ent_coef_optimizer is not None:
            saved_pytorch_variables = ["log_ent_coef"]
            state_dicts.append("ent_coef_optimizer")
        else:
            saved_pytorch_variables = ["ent_coef_tensor"]
        return state_dicts, saved_pytorch_variables

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 5000,
        tb_log_name: str = "MinOfflineSAC",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> OfflineAlgorithm:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )
