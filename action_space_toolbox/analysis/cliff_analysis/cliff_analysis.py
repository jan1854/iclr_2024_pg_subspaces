import dataclasses
import logging
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Literal

import filelock
import gym
import numpy as np
import omegaconf
import stable_baselines3
import stable_baselines3.common.buffers
import torch.multiprocessing
import yaml

from action_space_toolbox.analysis.analysis import Analysis
from action_space_toolbox.analysis.util import (
    evaluate_agent_returns,
    evaluate_returns_rollout_buffer,
    ReturnEvaluationResult,
    evaluate_agent_losses,
    LossEvaluationResult,
)
from action_space_toolbox.util.tensorboard_logs import TensorboardLogs
from sb3_utils.common.agent_spec import AgentSpec, HydraAgentSpec
from sb3_utils.common.buffer import fill_rollout_buffer
from sb3_utils.common.parameters import flatten_parameters
from sb3_utils.ppo.ppo_gradient import ppo_gradient

logger = logging.getLogger(__name__)


class CliffAnalysis(Analysis):
    def __init__(
        self,
        analysis_run_id: str,
        env_factory: Callable[[], gym.Env],
        agent_spec: AgentSpec,
        run_dir: Path,
        num_samples_true_gradient: int,
        num_trials: int,
        num_episodes_reward_eval: int,
        num_env_steps_training: int,
        gradient_normalization: Literal["original", "l1", "l2"],
        cliff_test_distance: float,
        alternate_agent_cfg: Optional[str],
        algorithm_overrides: Dict[str, Sequence[Dict[str, Any]]],
    ):
        super().__init__(
            "cliff_analysis",
            analysis_run_id,
            env_factory,
            agent_spec,
            run_dir,
        )
        self.num_samples_true_gradient = num_samples_true_gradient
        self.num_trials = num_trials
        self.num_episodes_reward_eval = num_episodes_reward_eval
        self.num_env_steps_training = num_env_steps_training
        self.gradient_normalization = gradient_normalization
        self.cliff_test_distance = cliff_test_distance
        self.results_dir = run_dir / "analyses" / "cliff_analysis" / analysis_run_id
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.results_file = self.results_dir / "results.yaml"
        self.results_file.touch()
        if alternate_agent_cfg is not None:
            # TODO: Bit ugly
            self.alternate_agent_cfg = omegaconf.OmegaConf.load(
                Path(__file__).parents[2]
                / "scripts"
                / "conf"
                / "algorithm"
                / (alternate_agent_cfg + ".yaml")
            )
            assert self.alternate_agent_cfg.name != "ppo"
            alternate_agent_cfg_dump_path = Path("alternate_agent_cfg.yaml")
            # Dump the alternate_agent_cfg (due to the hack above this is not done automatically by hydra)
            if not alternate_agent_cfg_dump_path.exists():
                omegaconf.OmegaConf.save(
                    self.alternate_agent_cfg, alternate_agent_cfg_dump_path
                )
        else:
            self.alternate_agent_cfg = None
        self.algorithm_overrides = algorithm_overrides

    def _do_analysis(
        self,
        env_step: int,
        logs: TensorboardLogs,
        overwrite_results: bool,
        verbose: bool,
    ) -> TensorboardLogs:

        agent = self.agent_spec.create_agent(self.env_factory())
        rollout_buffer_true_loss = stable_baselines3.common.buffers.RolloutBuffer(
            self.num_samples_true_gradient,
            agent.observation_space,
            agent.action_space,
            agent.device,
            agent.gae_lambda,
            agent.gamma,
        )
        rollout_buffer_true_loss_no_value_bootstrap = (
            stable_baselines3.common.buffers.RolloutBuffer(
                self.num_samples_true_gradient,
                agent.observation_space,
                agent.action_space,
                agent.device,
                agent.gae_lambda,
                agent.gamma,
            )
        )
        last_episode_done = fill_rollout_buffer(
            self.env_factory,
            self.agent_spec,
            rollout_buffer_true_loss,
            rollout_buffer_true_loss_no_value_bootstrap,
        )

        results = self._read_results_file()

        agent_names = ["ppo"]
        agent_specs = [self.agent_spec]
        if self.alternate_agent_cfg is not None:
            agent_names.append(self.alternate_agent_cfg.name)
            agent_specs.append(
                HydraAgentSpec(
                    self.alternate_agent_cfg,
                    self.agent_spec.device,
                    self.env_factory,
                    self.agent_spec.checkpoint_path,
                    agent_kwargs={"tensorboard_log": None},
                )
            )

        if overwrite_results or not self._check_logs_complete(results, env_step):
            gradient, _, _ = ppo_gradient(agent, next(rollout_buffer_true_loss.get()))
            if self.gradient_normalization == "original":
                # This is not the L1 norm (missing the abs), not sure why the Cliff Diving authors use this
                # normalization
                grad_sum = torch.sum(flatten_parameters(gradient))
                normalized_gradient = [g / grad_sum for g in gradient]
            elif self.gradient_normalization == "l1":
                grad_sum = torch.linalg.vector_norm(flatten_parameters(gradient), ord=1)
                normalized_gradient = [g / grad_sum for g in gradient]
            else:
                gradient_norm = torch.linalg.vector_norm(
                    flatten_parameters(gradient), ord=2
                )
                normalized_gradient = [g / gradient_norm for g in gradient]

            reward_checkpoint = evaluate_returns_rollout_buffer(
                rollout_buffer_true_loss_no_value_bootstrap,
                agent.gamma,
                last_episode_done,
            )

            losses_checkpoint = {
                name: evaluate_agent_losses(agent_spec, rollout_buffer_true_loss)
                for name, agent_spec in zip(agent_names, agent_specs)
            }

            cliff_test_parameters = [
                p - self.cliff_test_distance * g
                for p, g in zip(agent.policy.parameters(), normalized_gradient)
            ]
            reward_cliff_test = evaluate_agent_returns(
                self.agent_spec.copy_with_new_parameters(cliff_test_parameters),
                self.env_factory,
                num_episodes=self.num_episodes_reward_eval,
            )
            losses_cliff_test = {
                name: evaluate_agent_losses(
                    agent_spec.copy_with_new_parameters(cliff_test_parameters),
                    rollout_buffer_true_loss,
                )
                for name, agent_spec in zip(agent_names, agent_specs)
            }

            self.dump_pre_update_results(
                env_step,
                reward_checkpoint,
                losses_checkpoint,
                reward_cliff_test,
                losses_cliff_test,
                overwrite_results,
            )

        if env_step not in results:
            results[env_step] = {}
        curr_results = results[env_step]

        for name, agent_spec in zip(agent_names, agent_specs):
            if name not in curr_results:
                curr_results[name] = {}
            curr_results = curr_results[name]
            overrides = [{}] + list(
                omegaconf.OmegaConf.to_container(self.algorithm_overrides[name])
            )
            for curr_overrides in overrides:
                if overwrite_results or self._overrides_to_str(
                    curr_overrides
                ) not in curr_results.get(name, {}).get("configs", {}):
                    (
                        loss_after_update,
                        reward_after_update,
                    ) = self.evaluate_agent_overrides(
                        agent_spec,
                        curr_overrides,
                        rollout_buffer_true_loss,
                    )

                    self.dump_update_results(
                        env_step,
                        reward_after_update,
                        loss_after_update,
                        name,
                        curr_overrides,
                    )

        return TensorboardLogs()

    def evaluate_agent_overrides(
        self,
        agent_spec: AgentSpec,
        overrides: Dict[str, Any],
        rollout_buffer_true_loss: stable_baselines3.common.buffers.RolloutBuffer,
    ) -> Tuple[LossEvaluationResult, ReturnEvaluationResult]:
        agent_spec_overrides = agent_spec.copy_with_new_parameters(
            agent_kwargs=agent_spec.agent_kwargs | overrides
        )

        agent_specs_after_update = []
        for _ in range(self.num_trials):
            agent = agent_spec_overrides.create_agent(self.env_factory())
            agent.learn(self.num_env_steps_training, reset_num_timesteps=False)
            agent_specs_after_update.append(
                agent_spec_overrides.copy_with_new_parameters(
                    weights=list(agent.policy.parameters())
                )
            )

        loss_after_update = evaluate_agent_losses(
            agent_specs_after_update, rollout_buffer_true_loss
        )
        reward_after_update = evaluate_agent_returns(
            agent_specs_after_update,
            self.env_factory,
            num_episodes=self.num_episodes_reward_eval,
        )

        return loss_after_update, reward_after_update

    def dump_pre_update_results(
        self,
        env_step: int,
        reward_checkpoint: ReturnEvaluationResult,
        losses_checkpoint: Dict[str, LossEvaluationResult],
        reward_cliff_test: ReturnEvaluationResult,
        losses_cliff_test: Dict[str, LossEvaluationResult],
        override: bool,
    ):
        lock = filelock.FileLock(
            self.results_file.with_suffix(self.results_file.suffix + ".lock")
        )
        with lock.acquire(timeout=60):
            results = self._read_results_file()
            if env_step not in results:
                results[env_step] = {}
            curr_results = results[env_step]
            assert override or (
                "reward_checkpoint" not in curr_results
                and "losses_checkpoint" not in curr_results
            ), (
                f"Override not set but checkpoint results are non-empty "
                f"(step: {env_step}, results file: {self.results_file})."
            )

            curr_results["reward_checkpoint"] = {
                k: np.mean(v).item()
                for k, v in dataclasses.asdict(reward_checkpoint).items()
            }
            curr_results["losses_checkpoint"] = {
                name: {
                    k: np.mean(v).item()
                    for k, v in dataclasses.asdict(loss).items()
                    if len(v) > 0
                }
                for name, loss in losses_checkpoint.items()
            }
            curr_results["reward_cliff_test"] = {
                k: np.mean(v).item()
                for k, v in dataclasses.asdict(reward_cliff_test).items()
            }
            curr_results["losses_cliff_test"] = {
                name: {
                    k: np.mean(v).item()
                    for k, v in dataclasses.asdict(loss).items()
                    if len(v) > 0
                }
                for name, loss in losses_cliff_test.items()
            }
            with self.results_file.open("w") as f:
                yaml.dump(results, f)

    def dump_update_results(
        self,
        env_step: int,
        reward_after_update: ReturnEvaluationResult,
        loss_after_update: LossEvaluationResult,
        algorithm_name: str,
        hyperparameter_overrides: Dict[str, Any],
    ) -> None:
        lock = filelock.FileLock(
            self.results_file.with_suffix(self.results_file.suffix + ".lock")
        )
        with lock.acquire(timeout=60):
            results = self._read_results_file()
            if env_step not in results:
                results[env_step] = {}
            curr_results = results[env_step]
            if "configs" not in curr_results:
                curr_results["configs"] = {}
            curr_results = curr_results["configs"]
            if algorithm_name not in curr_results:
                curr_results[algorithm_name] = {}
            curr_results = curr_results[algorithm_name]
            overrides_str = self._overrides_to_str(hyperparameter_overrides)
            if overrides_str not in curr_results:
                curr_results[overrides_str] = {}
            curr_results = curr_results[overrides_str]
            curr_results["reward_update"] = {
                k: v.tolist()
                for k, v in dataclasses.asdict(reward_after_update).items()
            }
            curr_results["loss_update"] = {
                k: v.tolist()
                for k, v in dataclasses.asdict(loss_after_update).items()
                if len(v) > 0
            }
            with self.results_file.open("w") as f:
                yaml.dump(results, f)

    def _read_results_file(self) -> Dict[int, Dict[str, Any]]:
        with self.results_file.open("r") as f:
            results = yaml.safe_load(f)
        if results is None:
            results = {}
        return results

    @classmethod
    def _overrides_to_str(cls, overrides: Dict[str, Any]) -> str:
        if len(overrides) != 0:
            return ",".join([f"{k}={v}" for k, v in overrides.items()])
        else:
            return "default"

    @classmethod
    def _str_to_overrides(cls, overrides_str: str) -> Dict[str, Any]:
        if overrides_str == "default":
            return {}
        else:
            return {
                override[: override.find("=")]: override[override.find("=") + 1 :]
                for override in overrides_str.split(",")
            }

    @classmethod
    def _check_logs_complete(
        cls, results: Dict[int, Dict[str, Any]], env_step: int
    ) -> bool:
        if env_step in results:
            curr_results = results[env_step]
            return (
                "reward_checkpoint" in curr_results
                and "losses_checkpoint" in curr_results
                and "losses_cliff_test" in curr_results
                and "reward_cliff_test" in curr_results
            )
        else:
            return False
