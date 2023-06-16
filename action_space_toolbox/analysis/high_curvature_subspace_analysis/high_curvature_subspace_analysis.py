import logging
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional, Sequence

import gym
import numpy as np
import stable_baselines3
import stable_baselines3.common.buffers
import torch
from matplotlib import pyplot as plt

from action_space_toolbox.analysis.analysis import Analysis
from action_space_toolbox.analysis.hessian.hessian_eigen_cached_calculator import (
    HessianEigenCachedCalculator,
)
from action_space_toolbox.analysis.util import flatten_parameters, project
from action_space_toolbox.util.agent_spec import AgentSpec
from action_space_toolbox.util.sb3_training import (
    fill_rollout_buffer,
    ppo_gradient,
    sample_update_trajectory,
)
from action_space_toolbox.util.tensorboard_logs import TensorboardLogs


logger = logging.Logger(__name__)


class HighCurvatureSubspaceAnalysis(Analysis):
    def __init__(
        self,
        analysis_run_id: str,
        env_factory: Callable[[], gym.Env],
        agent_spec: AgentSpec,
        run_dir: Path,
        num_samples_true_loss: int,
        top_eigenvec_levels: Sequence[int],
        eigenvec_overlap_checkpoints: Sequence[int],
        overwrite_cached_eigen: bool,
    ):
        super().__init__(
            "high_curvature_subspace_analysis",
            analysis_run_id,
            env_factory,
            agent_spec,
            run_dir,
        )
        self.num_samples_true_loss = num_samples_true_loss
        self.top_eigenvec_levels = top_eigenvec_levels
        self.eigenvec_overlap_checkpoints = eigenvec_overlap_checkpoints
        self.overwrite_cached_eigen = overwrite_cached_eigen
        self.results_dir = run_dir / "analyses" / self.analysis_name / analysis_run_id
        self.results_dir.mkdir(exist_ok=True, parents=True)
        self.eigenspectrum_dir = self.results_dir / "eigenspectrum"
        self.eigenspectrum_dir.mkdir(exist_ok=True)

    def _do_analysis(
        self,
        env_step: int,
        logs: TensorboardLogs,
        overwrite_results: bool,
        verbose: bool,
    ) -> TensorboardLogs:
        agent = self.agent_spec.create_agent(self.env_factory())
        rollout_buffer_true_loss = stable_baselines3.common.buffers.RolloutBuffer(
            self.num_samples_true_loss,
            agent.observation_space,
            agent.action_space,
            agent.device,
            agent.gae_lambda,
            agent.gamma,
        )
        fill_rollout_buffer(
            self.env_factory,
            self.agent_spec,
            rollout_buffer_true_loss,
        )

        hess_eigen_calculator = HessianEigenCachedCalculator(self.run_dir)
        eigenvalues, eigenvectors = hess_eigen_calculator.get_eigen_combined_loss(
            agent,
            next(rollout_buffer_true_loss.get()),
            env_step,
            overwrite_cache=self.overwrite_cached_eigen,
        )
        (eigenvalues_policy, eigenvectors_policy), (
            eigenvalues_vf,
            eigenvectors_vf,
        ) = hess_eigen_calculator.get_eigen_policy_vf_loss(
            agent,
            next(rollout_buffer_true_loss.get()),
            env_step,
            overwrite_cache=self.overwrite_cached_eigen,
        )
        self._plot_eigenspectrum(
            eigenvalues, env_step, "combined loss", "combined_loss"
        )
        self._plot_eigenspectrum(
            eigenvalues_policy, env_step, "policy loss", "policy_loss"
        )
        self._plot_eigenspectrum(
            eigenvalues_vf, env_step, "value function", "value_function_loss"
        )

        rollout_buffer_gradient_estimates = (
            stable_baselines3.common.buffers.RolloutBuffer(
                agent.n_steps * agent.n_envs,
                agent.observation_space,
                agent.action_space,
                agent.device,
            )
        )
        fill_rollout_buffer(self.env_factory, agent, rollout_buffer_gradient_estimates)

        subspace_fracs_est_grad = self._calculate_gradient_subspace_fraction(
            eigenvectors, rollout_buffer_gradient_estimates.get()
        )
        keys = []
        for num_evs, subspace_frac in subspace_fracs_est_grad.items():
            curr_key = f"gradient_subspace_fraction_{num_evs:03d}evs/estimated_gradient"
            keys.append(curr_key)
            logs.add_scalar(curr_key, subspace_frac, env_step)
        logs.add_multiline_scalar(
            f"gradient_subspace_fraction/estimated_gradient", keys
        )
        subspace_fracs_true_grad = self._calculate_gradient_subspace_fraction(
            eigenvectors, rollout_buffer_true_loss.get()
        )
        keys = []
        for num_evs, subspace_frac in subspace_fracs_true_grad.items():
            curr_key = f"gradient_subspace_fraction_{num_evs:03d}evs/true_gradient"
            keys.append(curr_key)
            logs.add_scalar(curr_key, subspace_frac, env_step)
        logs.add_multiline_scalar(f"gradient_subspace_fraction/true_gradient", keys)

        overlaps = self._calculate_overlaps()
        for k, overlaps_top_k in overlaps.items():
            keys = []
            for t1, overlaps_t1 in overlaps_top_k.items():
                if len(overlaps_t1) > 0:
                    curr_key = f"overlaps_top{k:03d}_checkpoint{t1:07d}"
                    keys.append(curr_key)
                    for t2, overlap in overlaps_t1.items():
                        logs.add_scalar(curr_key, overlap, t2)
            logs.add_multiline_scalar(f"overlaps_top{k:03d}", keys)

        update_trajectory = sample_update_trajectory(
            self.agent_spec,
            rollout_buffer_gradient_estimates,
            agent.batch_size,
            n_epochs=agent.n_epochs,
        )
        agent_spec_after_update = self.agent_spec.copy_with_new_parameters(
            update_trajectory[-1]
        )
        _, eigenvectors_after_update = hess_eigen_calculator.get_eigen_combined_loss(
            agent_spec_after_update.create_agent(self.env_factory()),
            next(rollout_buffer_true_loss.get()),
            env_step,
            len(update_trajectory),
            overwrite_cache=self.overwrite_cached_eigen,
        )

        keys_update = []
        for k in self.top_eigenvec_levels:
            overlap = self._calculate_eigenvectors_overlap(
                eigenvectors[:, :k],
                eigenvectors_after_update[:, :k],
            )
            curr_key = f"overlaps_update_top{k:03d}"
            logs.add_scalar(curr_key, overlap, env_step)
            keys_update.append(curr_key)
        logs.add_multiline_scalar(f"overlaps_update", keys_update)

        return logs

    @classmethod
    def _calculate_eigenvectors_overlap(
        cls, eigenvectors1: torch.Tensor, eigenvectors2: torch.Tensor
    ) -> float:
        projected_evs = project(eigenvectors2, eigenvectors1, result_in_orig_space=True)
        return torch.mean(torch.norm(projected_evs, dim=0) ** 2).item()

    def _plot_eigenspectrum(
        self, eigenvalues: torch.Tensor, env_step: int, title: str, directory_name: str
    ) -> None:
        plt.title(f"Spectrum of the Hessian eigenvalues ({title})")
        plt.scatter(list(reversed(range(len(eigenvalues)))), eigenvalues)
        plt.savefig(self.eigenspectrum_dir / directory_name / f"{env_step}.pdf")
        plt.close()
        plt.title(f"Spectrum of the positive Hessian eigenvalues ({title})")
        eigenvalues_pos = eigenvalues[eigenvalues > 0]
        plt.scatter(
            list(reversed(range(len(eigenvalues_pos)))),
            eigenvalues_pos,
        )
        plt.yscale("log")
        plt.savefig(
            self.eigenspectrum_dir / directory_name / f"logscale_{env_step}.pdf"
        )
        plt.close()

    def _calculate_gradient_subspace_fraction(
        self,
        eigenvectors: torch.Tensor,
        rollout_buffer_samples: Optional[
            Iterable[stable_baselines3.common.buffers.RolloutBufferSamples]
        ] = None,
    ) -> Dict[int, float]:
        agent = self.agent_spec.create_agent()
        subspace_fractions = {}
        for batch in rollout_buffer_samples:
            gradient, _, _ = ppo_gradient(agent, batch)
            gradient = flatten_parameters(gradient).unsqueeze(1)
            for num_eigenvecs in self.top_eigenvec_levels:
                gradient_projected = project(
                    gradient,
                    eigenvectors[:, :num_eigenvecs],
                    result_in_orig_space=True,
                )
                sub_frac = (
                    (gradient_projected.T @ gradient_projected)
                    / (gradient.T @ gradient)
                ).item()
                if num_eigenvecs not in subspace_fractions:
                    subspace_fractions[num_eigenvecs] = []
                subspace_fractions[num_eigenvecs].append(sub_frac)
        return {
            num_evs: np.mean(sub_frac_list)
            for num_evs, sub_frac_list in subspace_fractions.items()
        }

    def _calculate_overlaps(self):
        hess_eigen_calculator = HessianEigenCachedCalculator(self.run_dir)
        start_checkpoints_eigenvecs = {
            num_eigenvecs: {} for num_eigenvecs in self.top_eigenvec_levels
        }
        overlaps = {num_eigenvecs: {} for num_eigenvecs in self.top_eigenvec_levels}
        for env_step, _, eigenvecs in hess_eigen_calculator.iter_cached_eigen(
            self.agent_spec.create_agent()
        ):
            for num_eigenvecs in self.top_eigenvec_levels:
                curr_start_checkpoints_eigenvecs = start_checkpoints_eigenvecs[
                    num_eigenvecs
                ]
                curr_overlaps = overlaps[num_eigenvecs]
                for (
                    checkpoint_env_step,
                    checkpoint_evs,
                ) in curr_start_checkpoints_eigenvecs.items():
                    curr_overlaps[checkpoint_env_step][
                        env_step
                    ] = self._calculate_eigenvectors_overlap(
                        checkpoint_evs[:, :num_eigenvecs],
                        eigenvecs[:, :num_eigenvecs],
                    )
                if env_step in self.eigenvec_overlap_checkpoints:
                    curr_start_checkpoints_eigenvecs[env_step] = eigenvecs
                    curr_overlaps[env_step] = {}
        return overlaps
