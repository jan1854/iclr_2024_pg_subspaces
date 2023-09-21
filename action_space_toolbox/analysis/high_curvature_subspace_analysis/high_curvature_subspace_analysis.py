import logging
from pathlib import Path
from typing import Callable, Dict, List, Literal, Optional, Sequence, Tuple

import gym
import numpy as np
import stable_baselines3
import stable_baselines3.common.buffers
import stable_baselines3.common.off_policy_algorithm
import stable_baselines3.common.on_policy_algorithm
import torch
from matplotlib import pyplot as plt
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.type_aliases import (
    RolloutBufferSamples,
    ReplayBufferSamples,
    TrainFreq,
    TrainFrequencyUnit,
)
from stable_baselines3.common.vec_env import DummyVecEnv

from action_space_toolbox.analysis.analysis import Analysis
from action_space_toolbox.analysis.hessian.hessian_eigen_cached_calculator import (
    HessianEigenCachedCalculator,
)
from action_space_toolbox.util.tensorboard_logs import TensorboardLogs
from sb3_utils.common.agent_spec import AgentSpec
from sb3_utils.common.buffer import fill_rollout_buffer, concatenate_buffer_samples
from sb3_utils.common.parameters import flatten_parameters, project_orthonormal
from sb3_utils.hessian.eigen.hessian_eigen import HessianEigen
from sb3_utils.common.loss import actor_critic_gradient
from sb3_utils.common.parameters import combine_actor_critic_parameter_vectors

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
        hessian_eigen: HessianEigen,
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
        self.hessian_eigen = hessian_eigen
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
        if isinstance(
            agent, stable_baselines3.common.on_policy_algorithm.OnPolicyAlgorithm
        ):
            (
                data_true_loss,
                data_estimated_loss,
            ) = self._collect_data_on_policy_algorithm(agent)
        else:
            (
                data_true_loss,
                data_estimated_loss,
            ) = self._collect_data_off_policy_algorithm(agent)

        hess_eigen_calculator = HessianEigenCachedCalculator(
            self.run_dir,
            self.hessian_eigen,
            max(self.top_eigenvec_levels),
            device=agent.device,
        )
        (
            eigenvals_combined,
            eigenvecs_combined,
        ) = hess_eigen_calculator.get_eigen_combined_loss(
            agent,
            data_true_loss,
            env_step,
            overwrite_cache=self.overwrite_cached_eigen,
        )
        (eigenvals_policy, eigenvecs_policy), (
            eigenvals_vf,
            eigenvecs_vf,
        ) = hess_eigen_calculator.get_eigen_policy_vf_loss(
            agent,
            data_true_loss,
            env_step,
            overwrite_cache=self.overwrite_cached_eigen,
        )

        data_estimated_loss_one_batch = concatenate_buffer_samples(data_estimated_loss)
        (
            (eigenvals_combined_low_samples, eigenvecs_combined_low_samples),
            (eigenvals_policy_low_samples, eigenvecs_policy_low_samples),
            (
                eigenvals_vf_low_samples,
                eigenvecs_vf_low_samples,
            ),
        ) = self.calculate_eigen(
            agent,
            data_estimated_loss_one_batch,
            self.top_eigenvec_levels[-1],
        )

        self._plot_eigenspectrum(eigenvals_policy, eigenvals_vf, env_step)

        loss_names = ["combined_loss", "policy_loss", "value_function_loss"]
        gradient_funcs = [
            lambda batch: actor_critic_gradient(
                agent, batch, all_gradients_fullsize=True
            )[0],
            lambda batch: actor_critic_gradient(
                agent, batch, all_gradients_fullsize=True
            )[1],
            lambda batch: actor_critic_gradient(
                agent, batch, all_gradients_fullsize=True
            )[2],
        ]
        eigenvals_params = [eigenvals_combined, eigenvals_policy, eigenvals_vf]
        eigenvecs_params = [eigenvecs_combined, eigenvecs_policy, eigenvecs_vf]
        eigenvecs_params_ls = [
            eigenvecs_combined_low_samples,
            eigenvecs_policy_low_samples,
            eigenvecs_vf_low_samples,
        ]
        for (loss_name, gradient_func, eigenvals, eigenvecs, eigenvecs_ls) in zip(
            loss_names,
            gradient_funcs,
            eigenvals_params,
            eigenvecs_params,
            eigenvecs_params_ls,
        ):
            subspace_fracs_est_grad = self._calculate_gradient_subspace_fraction(
                eigenvecs, gradient_func, data_estimated_loss
            )
            subspace_fracs_true_grad = self._calculate_gradient_subspace_fraction(
                eigenvecs, gradient_func, [data_true_loss]
            )
            subspace_fracs_est_grad_ls = self._calculate_gradient_subspace_fraction(
                eigenvecs_ls, gradient_func, data_estimated_loss
            )
            subspace_fracs_true_grad_ls = self._calculate_gradient_subspace_fraction(
                eigenvecs_ls, gradient_func, [data_true_loss]
            )
            self.log_subspace_metrics(
                env_step,
                subspace_fracs_est_grad,
                subspace_fracs_true_grad,
                loss_name,
                agent,
                logs,
            )
            self.log_subspace_metrics(
                env_step,
                subspace_fracs_est_grad_ls,
                subspace_fracs_true_grad_ls,
                loss_name,
                agent,
                logs,
                "low_sample",
            )

        return logs

    def log_subspace_metrics(
        self,
        env_step: int,
        subspace_fracs_est: Dict[int, float],
        subspace_fracs_true: Dict[int, float],
        loss_name: Literal["combined_loss", "policy_loss", "value_function_loss"],
        agent: stable_baselines3.common.base_class.BaseAlgorithm,
        logs: TensorboardLogs,
        prefix: Optional[str] = None,
    ):
        if prefix is None:
            prefix = ""
        else:
            prefix = prefix + "/"
        keys = []
        for num_evs, subspace_frac in subspace_fracs_est.items():
            curr_key = f"{prefix}gradient_subspace_fraction_{num_evs:03d}evs/estimated_gradient/{loss_name}"
            keys.append(curr_key)
            logs.add_scalar(curr_key, subspace_frac, env_step)
        logs.add_multiline_scalar(
            f"{prefix}gradient_subspace_fraction/estimated_gradient/{loss_name}", keys
        )
        keys = []
        for num_evs, subspace_frac in subspace_fracs_true.items():
            curr_key = f"{prefix}gradient_subspace_fraction_{num_evs:03d}evs/true_gradient/{loss_name}"
            keys.append(curr_key)
            logs.add_scalar(curr_key, subspace_frac, env_step)
        logs.add_multiline_scalar(
            f"{prefix}gradient_subspace_fraction/true_gradient/{loss_name}", keys
        )

        overlaps = self._calculate_overlaps(loss_name, agent)
        for k, overlaps_top_k in overlaps.items():
            keys = []
            for t1, overlaps_t1 in overlaps_top_k.items():
                if len(overlaps_t1) > 0:
                    curr_key = (
                        f"{prefix}overlaps_top{k:03d}_checkpoint{t1:07d}/{loss_name}"
                    )
                    keys.append(curr_key)
                    for t2, overlap in overlaps_t1.items():
                        logs.add_scalar(curr_key, overlap, t2)
            logs.add_multiline_scalar(f"{prefix}overlaps_top{k:03d}/{loss_name}", keys)

    @classmethod
    def _calculate_eigenvectors_overlap(
        cls, eigenvectors1: torch.Tensor, eigenvectors2: torch.Tensor
    ) -> float:
        projected_evs = project_orthonormal(
            eigenvectors2, eigenvectors1, result_in_orig_space=True
        )
        return torch.mean(torch.norm(projected_evs, dim=0) ** 2).item()

    def _plot_eigenspectrum(
        self,
        eigenvalues_policy: torch.Tensor,
        eigenvalues_vf: torch.Tensor,
        env_step: int,
    ) -> None:
        plt.title(f"Spectrum of the Hessian eigenvalues (policy loss)")
        plt.scatter(
            list(reversed(range(len(eigenvalues_policy)))),
            eigenvalues_policy.cpu().numpy(),
        )
        plt.savefig(self.eigenspectrum_dir / "policy_loss" / f"{env_step}.pdf")
        plt.close()
        plt.title(f"Spectrum of the Hessian eigenvalues (value function loss)")
        plt.scatter(
            list(reversed(range(len(eigenvalues_vf)))), eigenvalues_vf.cpu().numpy()
        )
        plt.savefig(self.eigenspectrum_dir / "value_function_loss" / f"{env_step}.pdf")
        plt.close()
        indices_pol = []
        indices_vf = []
        pol_idx = 0
        vf_idx = 0
        num_eigenvals = len(eigenvalues_policy) + len(eigenvalues_vf)
        while pol_idx < len(eigenvalues_policy) or vf_idx < len(eigenvalues_vf):
            if pol_idx < len(eigenvalues_policy) and (
                vf_idx >= len(eigenvalues_vf)
                or eigenvalues_policy[pol_idx] > eigenvalues_vf[vf_idx]
            ):
                indices_pol.append(num_eigenvals - 1 - (pol_idx + vf_idx))
                pol_idx += 1
            else:
                indices_vf.append(num_eigenvals - 1 - (pol_idx + vf_idx))
                vf_idx += 1

        plt.title(f"Spectrum of the Hessian eigenvalues (combined loss)")
        plt.scatter(
            indices_pol,
            eigenvalues_policy.cpu().numpy(),
            label="Policy eigenvalues",
            marker="x",
            s=10,
        )
        plt.scatter(
            indices_vf,
            eigenvalues_vf.cpu().numpy(),
            label="Value function eigenvalues",
            marker="x",
            s=10,
        )
        plt.legend()
        plt.savefig(self.eigenspectrum_dir / "combined_loss" / f"{env_step}.pdf")
        plt.close()

    def _calculate_gradient_subspace_fraction(
        self,
        eigenvectors: torch.Tensor,
        gradient_func: Callable[
            [stable_baselines3.common.buffers.RolloutBufferSamples],
            Sequence[torch.Tensor],
        ],
        rollout_buffer_samples: Optional[
            Sequence[stable_baselines3.common.buffers.RolloutBufferSamples]
        ] = None,
    ) -> Dict[int, float]:
        subspace_fractions = {}
        for batch in rollout_buffer_samples:
            gradient = gradient_func(batch)
            gradient = flatten_parameters(gradient).unsqueeze(1)
            for num_eigenvecs in self.top_eigenvec_levels:
                gradient_projected = project_orthonormal(
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
        self,
        loss_name: Literal["combined_loss", "policy_loss", "value_function_loss"],
        agent: stable_baselines3.common.base_class.BaseAlgorithm,
    ) -> Dict[int, Dict[int, Dict[int, float]]]:
        hess_eigen_calculator = HessianEigenCachedCalculator(
            self.run_dir, self.hessian_eigen
        )
        start_checkpoints_eigenvecs = {
            num_eigenvecs: {} for num_eigenvecs in self.top_eigenvec_levels
        }
        overlaps = {num_eigenvecs: {} for num_eigenvecs in self.top_eigenvec_levels}
        for env_step, _, eigenvecs in hess_eigen_calculator.iter_cached_eigen(
            agent, loss_name=loss_name
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
                    curr_overlaps[env_step] = {env_step: 1.0}
        return overlaps

    # TODO: This is redundant with HessianEigenCachedCalculator
    def calculate_eigen(
        self,
        agent: stable_baselines3.ppo.PPO,
        data: RolloutBufferSamples,
        num_eigenvectors: Optional[int],
    ) -> Tuple[
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor],
    ]:
        # Check that the there is no parameter sharing between policy and value function
        assert (
            sum(["shared" in name for name, _ in agent.policy.named_parameters()]) == 0
        )
        eigen = self.hessian_eigen.calculate_top_eigen(
            agent, data, num_eigenvectors, eigenvectors_fullsize=False
        )
        policy_eigenvectors_all_parameters = combine_actor_critic_parameter_vectors(
            eigen.policy.eigenvectors, None, agent
        )
        vf_eigenvectors_all_parameters = combine_actor_critic_parameter_vectors(
            None, eigen.value_function.eigenvectors, agent
        )
        (
            eigenvals_combined,
            eigenvecs_combined,
        ) = HessianEigenCachedCalculator.collect_top_eigenvectors(
            agent, eigen, num_eigenvectors
        )
        return (
            (eigenvals_combined[:100], eigenvecs_combined),
            (eigen.policy.eigenvalues, policy_eigenvectors_all_parameters),
            (
                eigen.value_function.eigenvalues,
                vf_eigenvectors_all_parameters,
            ),
        )

    def _collect_data_on_policy_algorithm(
        self,
        agent: stable_baselines3.common.on_policy_algorithm.OnPolicyAlgorithm,
    ) -> Tuple[RolloutBufferSamples, List[RolloutBufferSamples]]:
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

        return next(rollout_buffer_true_loss.get()), list(
            rollout_buffer_gradient_estimates.get(agent.batch_size)
        )

    def _collect_data_off_policy_algorithm(
        self,
        agent: stable_baselines3.common.off_policy_algorithm.OffPolicyAlgorithm,
    ) -> Tuple[ReplayBufferSamples, List[ReplayBufferSamples]]:
        # If the replay buffer is empty (step 0), collect some random data
        if not agent.replay_buffer.full and agent.replay_buffer.pos == 0:
            env = DummyVecEnv([lambda: self.env_factory()])
            agent._last_obs = env.reset()
            # collect_rollouts requires a callback for some reason
            callback = CallbackList([])
            callback.init_callback(agent)
            agent._total_timesteps = 50_000
            agent.collect_rollouts(
                env,
                callback,
                TrainFreq(50_000, TrainFrequencyUnit.STEP),
                agent.replay_buffer,
                agent.action_noise,
                50_000,
            )
        max_idx = (
            agent.replay_buffer.buffer_size
            if agent.replay_buffer.full
            else agent.replay_buffer.pos
        )
        # Set the number of batches for the estimated gradient so that the same amount of data is used as for
        # default PPO
        return agent.replay_buffer._get_samples(np.arange(max_idx)), [
            agent.replay_buffer.sample(agent.batch_size)
            for _ in range(2048 // agent.batch_size)
        ]
