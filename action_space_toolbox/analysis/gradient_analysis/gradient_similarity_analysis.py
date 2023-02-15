import math
from typing import Optional, Sequence, Tuple

import numpy as np
import stable_baselines3
import torch
from stable_baselines3.common.buffers import RolloutBuffer

from action_space_toolbox.util.sb3_training import ppo_gradient
from action_space_toolbox.util.tensorboard_logs import TensorboardLogs


class GradientSimilarityAnalysis:
    def __init__(
        self,
        max_num_gradient_estimates: int,
        batch_sizes_gradient_estimates: Sequence[int],
        analysis_run_id: str,
    ):
        self.max_num_gradient_estimates = max_num_gradient_estimates
        self.batch_sizes_gradient_estimates = np.asarray(batch_sizes_gradient_estimates)
        self.analysis_run_id = analysis_run_id

    def analyze(
        self,
        rollout_buffer_true_gradient: RolloutBuffer,
        rollout_buffer_estimates: RolloutBuffer,
        agent: stable_baselines3.ppo.PPO,
        env_step: int,
    ) -> TensorboardLogs:
        (
            policy_gradient_true,
            vf_gradient_true,
            combined_gradient_true,
        ) = self._gradients_from_buffer(rollout_buffer_true_gradient, agent)
        (
            policy_gradient_agent,
            vf_gradient_agent,
            combined_gradient_agent,
        ) = self._gradients_from_buffer(
            rollout_buffer_estimates,
            agent,
            agent.batch_size,
            self.max_num_gradient_estimates,
        )
        gradient_estimates_other = [
            self._gradients_from_buffer(
                rollout_buffer_estimates,
                agent,
                batch_size,
                self.max_num_gradient_estimates,
            )
            for batch_size in self.batch_sizes_gradient_estimates
        ]
        policy_gradients_other, vf_gradients_other, combined_gradients_other = zip(
            *gradient_estimates_other
        )

        logs = TensorboardLogs()
        self._evaluate_gradient_estimates(
            policy_gradient_true,
            policy_gradient_agent,
            policy_gradients_other,
            "policy",
            env_step,
            logs,
        )
        self._evaluate_gradient_estimates(
            vf_gradient_true,
            vf_gradient_agent,
            vf_gradients_other,
            "value_function",
            env_step,
            logs,
        )
        self._evaluate_gradient_estimates(
            combined_gradient_true,
            combined_gradient_agent,
            combined_gradients_other,
            "combined",
            env_step,
            logs,
        )
        return logs

    def _evaluate_gradient_estimates(
        self,
        true_gradient: torch.Tensor,
        estimated_gradient_agent: torch.Tensor,
        estimated_gradients_other: torch.Tensor,
        gradient_name: str,
        env_step: int,
        logs: TensorboardLogs,
    ) -> None:
        similarity_original_true = self.similarity_true_gradient(
            true_gradient, estimated_gradient_agent
        )
        similarity_original_estimates = self.similarity_estimated_gradients(
            estimated_gradient_agent
        )
        similarities_other_true = [
            self.similarity_true_gradient(true_gradient, gradient_estimates)
            for gradient_estimates in estimated_gradients_other
        ]
        similarities_other_estimates = [
            self.similarity_estimated_gradients(gradient_estimates)
            for gradient_estimates in estimated_gradients_other
        ]

        logs.add_scalar(
            f"gradient_analysis/{self.analysis_run_id}/{gradient_name}/similarity_estimates_true_gradient",
            similarity_original_true,
            env_step,
        )
        logs.add_scalar(
            f"gradient_analysis/{self.analysis_run_id}/{gradient_name}/similarity_gradient_estimates",
            similarity_original_estimates,
            env_step,
        )
        logs.add_step_plot(
            f"gradient_analysis_step_plots/{self.analysis_run_id}/{gradient_name}/"
            f"similarity_estimates_true_gradient_diff_batch_sizes_logx_{env_step:07d}",
            np.round(np.log10(self.batch_sizes_gradient_estimates)).astype(int),
            similarities_other_true,
        )
        logs.add_step_plot(
            f"gradient_analysis_step_plots/{self.analysis_run_id}/{gradient_name}/"
            f"similarity_gradient_estimates_diff_batch_sizes_logx_{env_step:07d}",
            np.round(np.log10(self.batch_sizes_gradient_estimates)).astype(int),
            similarities_other_estimates,
        )

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
        num_samples_per_gradient: Optional[int] = None,
        max_num_gradients: Optional[int] = None,
        max_batch_size: int = 100000,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        policy_gradients = []
        vf_gradients = []
        combined_gradients = []
        if num_samples_per_gradient is None:
            num_samples_per_gradient = rollout_buffer.buffer_size
        assert (
            num_samples_per_gradient < max_batch_size
            or num_samples_per_gradient % max_batch_size == 0
        ), "The maximum number of samples has to be divisible by the maximum batch size"
        batch_size = min(num_samples_per_gradient, max_batch_size)
        batches_per_gradient = num_samples_per_gradient // batch_size

        for i, batch in enumerate(rollout_buffer.get(batch_size)):
            # Do not use incomplete batches
            if (
                max_num_gradients is not None
                and i >= max_num_gradients * batches_per_gradient
            ) or (batch.actions.shape[0] != batch_size):
                break
            combined_gradient, policy_gradient, vf_gradient = ppo_gradient(agent, batch)
            policy_gradients.append(torch.cat([g.flatten() for g in policy_gradient]))
            vf_gradients.append(torch.cat([g.flatten() for g in vf_gradient]))
            combined_gradients.append(
                torch.cat([g.flatten() for g in combined_gradient])
            )

        policy_gradients = cls.average_gradient_batches(
            torch.stack(policy_gradients), batches_per_gradient
        )
        vf_gradients = cls.average_gradient_batches(
            torch.stack(vf_gradients), batches_per_gradient
        )
        combined_gradients = cls.average_gradient_batches(
            torch.stack(combined_gradients), batches_per_gradient
        )

        return policy_gradients, vf_gradients, combined_gradients

    @classmethod
    def average_gradient_batches(
        cls, gradient_batches: torch.Tensor, batches_per_gradient: int
    ) -> torch.Tensor:
        num_batches = gradient_batches.shape[0]
        # num_batches might not be divisible by batches_per_gradient
        # --> do not compute gradients for which we do not have enough batches
        num_gradients = math.floor(num_batches / batches_per_gradient)
        gradient_batches = gradient_batches[: num_gradients * batches_per_gradient]
        gradient_batches = gradient_batches.reshape(
            num_gradients, batches_per_gradient, gradient_batches.shape[-1]
        )
        return gradient_batches.mean(dim=1)

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
