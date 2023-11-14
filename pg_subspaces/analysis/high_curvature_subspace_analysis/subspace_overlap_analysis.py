from pathlib import Path
from typing import Callable, Dict, Literal, Sequence

import gym
import stable_baselines3.common.base_class
import torch

from pg_subspaces.analysis.analysis import Analysis
from pg_subspaces.analysis.hessian.hessian_eigen_cached_calculator import (
    HessianEigenCachedCalculator,
)
from pg_subspaces.metrics.tensorboard_logs import TensorboardLogs
from pg_subspaces.sb3_utils.common.agent_spec import AgentSpec
from pg_subspaces.sb3_utils.common.parameters import project_orthonormal
from pg_subspaces.sb3_utils.hessian.eigen.hessian_eigen import HessianEigen


class SubspaceOverlapAnalysis(Analysis):
    def __init__(
        self,
        analysis_run_id: str,
        env_factory_or_dataset: Callable[[], gym.Env],
        agent_spec: AgentSpec,
        run_dir: Path,
        top_eigenvec_levels: Sequence[int],
        hessian_eigen: HessianEigen,
        eigenvec_overlap_checkpoints: Sequence[int],
        lock_analysis_log_file: bool = True,
    ):
        super().__init__(
            "subspace_overlaps_analysis",
            analysis_run_id,
            env_factory_or_dataset,
            agent_spec,
            run_dir,
            lock_analysis_log_file,
        )
        self.top_eigenvec_levels = top_eigenvec_levels
        self.hessian_eigen = hessian_eigen
        self.eigenvec_overlap_checkpoints = eigenvec_overlap_checkpoints

    def _do_analysis(
        self,
        env_step: int,
        logs: TensorboardLogs,
        overwrite_results: bool,
        verbose: bool,
    ) -> TensorboardLogs:
        agent = self.agent_spec.create_agent()
        loss_names = ["combined_loss", "policy_loss", "value_function_loss"]
        prefixes = ["", "low_sample/"]
        for loss_name in loss_names:
            for prefix in prefixes:
                overlaps = self._calculate_overlaps(loss_name, agent)
                for k, overlaps_top_k in overlaps.items():
                    keys = []
                    for t1, overlaps_t1 in overlaps_top_k.items():
                        if len(overlaps_t1) > 0:
                            curr_key = f"{prefix}overlaps_top{k:03d}_checkpoint{t1:07d}/{loss_name}"
                            keys.append(curr_key)
                            for t2, overlap in overlaps_t1.items():
                                logs.add_scalar(curr_key, overlap, t2)
                    logs.add_multiline_scalar(
                        f"{prefix}overlaps_top{k:03d}/{loss_name}", keys
                    )
        return logs

    def _calculate_overlaps(
        self,
        loss_name: Literal["combined_loss", "policy_loss", "value_function_loss"],
        agent: stable_baselines3.common.base_class.BaseAlgorithm,
    ) -> Dict[int, Dict[int, Dict[int, float]]]:
        hess_eigen_calculator = HessianEigenCachedCalculator(
            self.run_dir, self.hessian_eigen, skip_cacheing=False
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

    @classmethod
    def _calculate_eigenvectors_overlap(
        cls, eigenvectors1: torch.Tensor, eigenvectors2: torch.Tensor
    ) -> float:
        projected_evs = project_orthonormal(
            eigenvectors2, eigenvectors1, result_in_orig_space=True
        )
        return torch.mean(torch.norm(projected_evs, dim=0) ** 2).item()
