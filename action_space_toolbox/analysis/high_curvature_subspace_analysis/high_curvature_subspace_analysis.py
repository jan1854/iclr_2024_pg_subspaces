import logging
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional, Sequence, Literal

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
        (self.eigenspectrum_dir / "combined_loss").mkdir(exist_ok=True)
        (self.eigenspectrum_dir / "policy_loss").mkdir(exist_ok=True)
        (self.eigenspectrum_dir / "value_function_loss").mkdir(exist_ok=True)

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

        rollout_buffer_gradient_estimates = (
            stable_baselines3.common.buffers.RolloutBuffer(
                agent.n_steps * agent.n_envs,
                agent.observation_space,
                agent.action_space,
                agent.device,
            )
        )
        fill_rollout_buffer(self.env_factory, agent, rollout_buffer_gradient_estimates)

        hess_eigen_calculator = HessianEigenCachedCalculator(
            self.run_dir, device=agent.device
        )
        (
            eigenvals_combined,
            eigenvecs_combined,
        ) = hess_eigen_calculator.get_eigen_combined_loss(
            agent,
            next(rollout_buffer_true_loss.get()),
            env_step,
            overwrite_cache=self.overwrite_cached_eigen,
        )
        (eigenvals_policy, eigenvecs_policy), (
            eigenvals_vf,
            eigenvecs_vf,
        ) = hess_eigen_calculator.get_eigen_policy_vf_loss(
            agent,
            next(rollout_buffer_true_loss.get()),
            env_step,
            overwrite_cache=self.overwrite_cached_eigen,
        )

        update_trajectory = sample_update_trajectory(
            self.agent_spec,
            rollout_buffer_gradient_estimates,
            agent.batch_size,
            n_epochs=agent.n_epochs,
        )
        agent_spec_after_update = self.agent_spec.copy_with_new_parameters(
            update_trajectory[-1]
        )
        (
            _,
            eigenvecs_after_update_combined,
        ) = hess_eigen_calculator.get_eigen_combined_loss(
            agent_spec_after_update.create_agent(self.env_factory()),
            next(rollout_buffer_true_loss.get()),
            env_step,
            len(update_trajectory),
            overwrite_cache=self.overwrite_cached_eigen,
        )
        (_, eigenvecs_after_update_policy,), (
            _,
            eigenvecs_after_update_vf,
        ) = hess_eigen_calculator.get_eigen_policy_vf_loss(
            agent_spec_after_update.create_agent(self.env_factory()),
            next(rollout_buffer_true_loss.get()),
            env_step,
            len(update_trajectory),
            overwrite_cache=self.overwrite_cached_eigen,
        )

        loss_names = ["combined_loss", "policy_loss", "value_function_loss"]
        gradient_funcs = [
            lambda batch: ppo_gradient(agent, batch, all_gradients_fullsize=True)[0],
            lambda batch: ppo_gradient(agent, batch, all_gradients_fullsize=True)[1],
            lambda batch: ppo_gradient(agent, batch, all_gradients_fullsize=True)[2],
        ]
        eigenvals_params = [eigenvals_combined, eigenvals_policy, eigenvals_vf]
        eigenvecs_params = [eigenvecs_combined, eigenvecs_policy, eigenvecs_vf]
        eigenvecs_after_update_params = [
            eigenvecs_after_update_combined,
            eigenvecs_after_update_policy,
            eigenvecs_after_update_vf,
        ]
        for (
            loss_name,
            gradient_func,
            eigenvals,
            eigenvecs,
            eigenvecs_after_update,
        ) in zip(
            loss_names,
            gradient_funcs,
            eigenvals_params,
            eigenvecs_params,
            eigenvecs_after_update_params,
        ):
            self._plot_eigenspectrum(
                eigenvals, env_step, loss_name, loss_name.replace("_", " ")
            )

            subspace_fracs_est_grad = self._calculate_gradient_subspace_fraction(
                eigenvecs,
                gradient_func,
                rollout_buffer_gradient_estimates.get(agent.batch_size),
            )
            keys = []
            for num_evs, subspace_frac in subspace_fracs_est_grad.items():
                curr_key = f"gradient_subspace_fraction_{num_evs:03d}evs/estimated_gradient/{loss_name}"
                keys.append(curr_key)
                logs.add_scalar(curr_key, subspace_frac, env_step)
            logs.add_multiline_scalar(
                f"gradient_subspace_fraction/estimated_gradient/{loss_name}", keys
            )
            subspace_fracs_true_grad = self._calculate_gradient_subspace_fraction(
                eigenvecs, gradient_func, rollout_buffer_true_loss.get()
            )
            keys = []
            for num_evs, subspace_frac in subspace_fracs_true_grad.items():
                curr_key = f"gradient_subspace_fraction_{num_evs:03d}evs/true_gradient/{loss_name}"
                keys.append(curr_key)
                logs.add_scalar(curr_key, subspace_frac, env_step)
            logs.add_multiline_scalar(
                f"gradient_subspace_fraction/true_gradient/{loss_name}", keys
            )

            overlaps = self._calculate_overlaps(loss_name)
            for k, overlaps_top_k in overlaps.items():
                keys = []
                for t1, overlaps_t1 in overlaps_top_k.items():
                    if len(overlaps_t1) > 0:
                        curr_key = f"overlaps_top{k:03d}_checkpoint{t1:07d}/{loss_name}"
                        keys.append(curr_key)
                        for t2, overlap in overlaps_t1.items():
                            logs.add_scalar(curr_key, overlap, t2)
                logs.add_multiline_scalar(f"overlaps_top{k:03d}/{loss_name}", keys)

            keys_update = []
            for k in self.top_eigenvec_levels:
                overlap = self._calculate_eigenvectors_overlap(
                    eigenvecs[:, :k],
                    eigenvecs_after_update[:, :k],
                )
                curr_key = f"overlaps_update_top{k:03d}/{loss_name}"
                logs.add_scalar(curr_key, overlap, env_step)
                keys_update.append(curr_key)
            logs.add_multiline_scalar(f"overlaps_update/{loss_name}", keys_update)

        return logs

    @classmethod
    def _calculate_eigenvectors_overlap(
        cls, eigenvectors1: torch.Tensor, eigenvectors2: torch.Tensor
    ) -> float:
        projected_evs = project(eigenvectors2, eigenvectors1, result_in_orig_space=True)
        return torch.mean(torch.norm(projected_evs, dim=0) ** 2).item()

    def _plot_eigenspectrum(
        self,
        eigenvalues: torch.Tensor,
        env_step: int,
        directory_name: str,
        loss_descr: str,
    ) -> None:
        plt.title(f"Spectrum of the Hessian eigenvalues ({loss_descr})")
        plt.scatter(list(reversed(range(len(eigenvalues)))), eigenvalues.cpu().numpy())
        plt.savefig(self.eigenspectrum_dir / directory_name / f"{env_step}.pdf")
        plt.close()
        plt.title(f"Spectrum of the positive Hessian eigenvalues ({loss_descr})")
        eigenvalues_pos = eigenvalues[eigenvalues > 0]
        plt.scatter(
            list(reversed(range(len(eigenvalues_pos)))),
            eigenvalues_pos.cpu().numpy(),
        )
        plt.yscale("log")
        plt.savefig(
            self.eigenspectrum_dir / directory_name / f"logscale_{env_step}.pdf"
        )
        plt.close()

    def _calculate_gradient_subspace_fraction(
        self,
        eigenvectors: torch.Tensor,
        gradient_func: Callable[
            [stable_baselines3.common.buffers.RolloutBufferSamples],
            Sequence[torch.Tensor],
        ],
        rollout_buffer_samples: Optional[
            Iterable[stable_baselines3.common.buffers.RolloutBufferSamples]
        ] = None,
    ) -> Dict[int, float]:
        subspace_fractions = {}
        for batch in rollout_buffer_samples:
            gradient = gradient_func(batch)
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

    def _calculate_overlaps(
        self, loss_name: Literal["combined_loss", "policy_loss", "value_function_loss"]
    ) -> Dict[int, Dict[int, Dict[int, float]]]:
        hess_eigen_calculator = HessianEigenCachedCalculator(self.run_dir)
        start_checkpoints_eigenvecs = {
            num_eigenvecs: {} for num_eigenvecs in self.top_eigenvec_levels
        }
        overlaps = {num_eigenvecs: {} for num_eigenvecs in self.top_eigenvec_levels}
        for env_step, _, eigenvecs in hess_eigen_calculator.iter_cached_eigen(
            self.agent_spec.create_agent(), loss_name=loss_name
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
