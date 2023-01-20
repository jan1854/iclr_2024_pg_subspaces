from typing import Optional, Sequence

import numpy as np
import stable_baselines3
import torch
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.type_aliases import RolloutBufferSamples

from action_space_toolbox.util.sb3_training import ppo_loss
from action_space_toolbox.util.tensorboard_logs import TensorboardLogs


class GradientSimilarityAnalysis:
    def __init__(
        self,
        max_num_gradient_estimates: int,
        batch_sizes_gradient_estimates: Sequence[int],
        analysis_run_id: str,
    ):
        self.max_num_gradient_estimates = max_num_gradient_estimates
        self.batch_sizes_gradient_estimates = np.array(batch_sizes_gradient_estimates)
        self.analysis_run_id = analysis_run_id

    def analyze(
        self,
        rollout_buffer_true_gradient: RolloutBuffer,
        rollout_buffer_estimates: RolloutBuffer,
        agent: stable_baselines3.ppo.PPO,
        env_step: int,
    ) -> TensorboardLogs:
        true_gradient = self._gradients_from_buffer(rollout_buffer_true_gradient, agent)
        gradient_estimates_original_batch_size = self._gradients_from_buffer(
            rollout_buffer_estimates,
            agent,
            agent.batch_size,
            self.max_num_gradient_estimates,
        )
        gradient_estimates_different_batch_sizes = [
            self._gradients_from_buffer(
                rollout_buffer_estimates,
                agent,
                batch_size,
                self.max_num_gradient_estimates,
            )
            for batch_size in self.batch_sizes_gradient_estimates
        ]

        similarity_original_true = self.similarity_true_gradient(
            true_gradient, gradient_estimates_original_batch_size
        )
        similarity_original_estimates = self.similarity_estimated_gradients(
            gradient_estimates_original_batch_size
        )
        similarities_other_true = [
            self.similarity_true_gradient(true_gradient, gradient_estimates)
            for gradient_estimates in gradient_estimates_different_batch_sizes
        ]
        similarities_other_estimates = [
            self.similarity_estimated_gradients(gradient_estimates)
            for gradient_estimates in gradient_estimates_different_batch_sizes
        ]

        logs = TensorboardLogs()
        logs.add_scalar(
            f"gradient_analysis/{self.analysis_run_id}/similarity_estimates_true_gradient",
            similarity_original_true,
            env_step,
        )
        logs.add_scalar(
            f"gradient_analysis/{self.analysis_run_id}/similarity_gradient_estimates",
            similarity_original_estimates,
            env_step,
        )
        logs.add_step_plot(
            f"gradient_analysis_step_plots/{self.analysis_run_id}/"
            f"similarity_estimates_true_gradient_diff_batch_sizes_logx_{env_step:07d}",
            np.log10(self.batch_sizes_gradient_estimates),
            similarities_other_true,
        )
        logs.add_step_plot(
            f"gradient_analysis_step_plots/{self.analysis_run_id}/"
            f"similarity_gradient_estimates_diff_batch_sizes_logx_{env_step:07d}",
            np.log10(self.batch_sizes_gradient_estimates),
            similarities_other_estimates,
        )
        return logs

    @classmethod
    def similarity_true_gradient(
        cls,
        true_gradient: torch.Tensor,
        estimated_gradients: torch.Tensor,
    ) -> float:

        cos_similarities = cls._cosine_similarities(true_gradient, estimated_gradients)
        return cos_similarities.mean().item()

    @classmethod
    def similarity_estimated_gradients(cls, estimated_gradients: torch.Tensor) -> float:
        dist_matrix = cls._cosine_similarities(estimated_gradients, estimated_gradients)

        # Calculate the average pairwise similarity
        dist_sum = (dist_matrix.sum() - torch.trace(dist_matrix)) / 2
        n = dist_matrix.shape[0]
        return (dist_sum / (n * (n - 1) / 2)).item()

    @classmethod
    def _gradients_from_buffer(
        cls,
        rollout_buffer: RolloutBuffer,
        agent: stable_baselines3.ppo.PPO,
        batch_size: Optional[int] = None,
        max_num_gradients: Optional[int] = None,
    ) -> torch.Tensor:
        gradients = []
        for i, batch in enumerate(rollout_buffer.get(batch_size)):
            # Do not use incomplete batches
            if (max_num_gradients is not None and i >= max_num_gradients) or (
                batch_size is not None and batch.actions.shape[0] != batch_size
            ):
                break
            gradients.append(cls._ppo_gradient(agent, batch))
        return torch.stack(gradients)

    @classmethod
    def _cosine_similarities(
        cls, t1: torch.Tensor, t2: torch.Tensor, eps: float = 1e-8
    ) -> torch.Tensor:
        t1_norm = t1.norm(dim=1)[:, None]
        t2_norm = t2.norm(dim=1)[:, None]
        t1_normalized = t1 / torch.clamp(t1_norm, min=eps)
        t2_normalized = t2 / torch.clamp(t2_norm, min=eps)
        sim_mt = torch.mm(t1_normalized, t2_normalized.transpose(0, 1))
        return sim_mt

    @classmethod
    def _ppo_gradient(
        cls, agent: stable_baselines3.ppo.PPO, rollout_data: RolloutBufferSamples
    ) -> torch.Tensor:
        # TODO: This current includes both the gradients for the policy and the value function, should it be this way?
        loss, _ = ppo_loss(agent, rollout_data)
        agent.policy.zero_grad()
        loss.backward()
        return torch.cat([p.grad.flatten() for p in agent.policy.parameters()])
