from pathlib import Path
from typing import Callable, List

import gym
import stable_baselines3
import stable_baselines3.common.buffers
from matplotlib import pyplot as plt

from action_space_toolbox.analysis.analysis import Analysis
from action_space_toolbox.analysis.hessian.hessian_eigen_cached_calculator import (
    HessianEigenCachedCalculator,
)
from action_space_toolbox.util.agent_spec import AgentSpec
from action_space_toolbox.util.sb3_training import fill_rollout_buffer
from action_space_toolbox.util.tensorboard_logs import TensorboardLogs


class HessianEigenspectrum(Analysis):
    def __init__(
        self,
        analysis_run_id: str,
        env_factory: Callable[[], gym.Env],
        agent_spec: AgentSpec,
        run_dir: Path,
        num_samples_true_loss: int,
        num_eigen_spectrum: int,
    ):
        super().__init__(
            "high_curvature_subspace_analysis",
            analysis_run_id,
            env_factory,
            agent_spec,
            run_dir,
        )
        self.num_samples_true_loss = num_samples_true_loss
        self.num_eigen_spectrum = num_eigen_spectrum
        self.results_dir = run_dir / "analyses" / "cliff_analysis" / analysis_run_id
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

        # TODO: Remove the low_budget_evs hack below
        hess_eigen_calc = HessianEigenCachedCalculator(
            self.run_dir / "low_budget_evs",
            self.num_eigen_spectrum,
            max_iter=100,
            tol=1e-3,
        )
        hess_eigen = hess_eigen_calc.get_eigen(
            agent, next(rollout_buffer_true_loss.get()), env_step, show_progress=True
        )
        self._plot_eigenspectrum(hess_eigen.eigenvalues, env_step)
        return logs

    def _plot_eigenspectrum(self, eigenvalues: List[float], env_step: int) -> None:
        eigenvalues = sorted(eigenvalues)
        plt.title(
            f"Spectrum of the top-{self.num_eigen_spectrum} (absolute value) eigenvalues."
        )
        plt.scatter(range(len(eigenvalues)), eigenvalues)
        plt.savefig(self.eigenspectrum_dir / f"{env_step}.pdf")
        plt.close()
