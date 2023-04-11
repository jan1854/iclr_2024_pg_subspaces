import dataclasses
import itertools
import logging
from pathlib import Path
from typing import Callable, Any, Dict, Sequence, Optional

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
    flatten_parameters,
    evaluate_agent_returns,
    evaluate_returns_rollout_buffer,
    ReturnEvaluationResult,
    evaluate_agent_losses,
    LossEvaluationResult,
)
from action_space_toolbox.util.agent_spec import AgentSpec
from action_space_toolbox.util.get_episode_length import get_episode_length
from action_space_toolbox.util.sb3_training import (
    ppo_gradient,
    fill_rollout_buffer,
)
from action_space_toolbox.util.tensorboard_logs import TensorboardLogs


logger = logging.getLogger(__name__)


class CliffAnalysis(Analysis):
    def __init__(
        self,
        analysis_run_id: str,
        env_factory: Callable[[], gym.Env],
        agent_spec: AgentSpec,
        run_dir: Path,
        num_processes: int,
        num_samples_true_gradient: int,
        num_steps_reward_eval: int,
        cliff_test_distance: float,
        cliff_reward_decrease: float,
        cliff_reward_decrease_global: float,
        a2c_cfg: Optional[str],
        algorithm_overrides: Dict[str, Sequence[Dict[str, Any]]],
    ):
        super().__init__(
            "cliff_analysis",
            analysis_run_id,
            env_factory,
            agent_spec,
            run_dir,
            num_processes,
        )
        self.num_samples_true_gradient = num_samples_true_gradient
        self.num_steps_reward_eval = num_steps_reward_eval
        self.cliff_test_distance = cliff_test_distance
        self.cliff_reward_decrease = cliff_reward_decrease
        self.cliff_reward_decrease_global = cliff_reward_decrease_global
        self.global_reward_range = 0.01  # TODO
        self.results_dir = run_dir / "analyses" / "cliff_analysis" / analysis_run_id
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.results_file = self.results_dir / "results.yaml"
        self.results_file.touch()
        if a2c_cfg is not None:
            # TODO: Bit ugly
            self.a2c_config = omegaconf.OmegaConf.load(
                Path(__file__).parents[2]
                / "scripts"
                / "conf"
                / "algorithm"
                / (a2c_cfg + ".yaml")
            )
        else:
            self.a2c_config = None
        self.algorithm_overrides = algorithm_overrides

    def _do_analysis(
        self,
        process_pool: torch.multiprocessing.Pool,
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
        fill_rollout_buffer(
            self.env_factory,
            self.agent_spec,
            rollout_buffer_true_loss,
            rollout_buffer_true_loss_no_value_bootstrap,
            num_spawned_processes=self.num_processes,
        )

        results = self._read_results_file()

        # TODO: Check whether env_step
        if env_step not in results:
            results[env_step] = {}
        curr_results = results[env_step]
        if overwrite_results or self._check_logs_complete(results, env_step):
            gradient, _, _ = ppo_gradient(agent, next(rollout_buffer_true_loss.get()))
            gradient_norm = torch.norm(flatten_parameters(gradient))
            normalized_gradient = [g / gradient_norm for g in gradient]

            reward_checkpoint = evaluate_returns_rollout_buffer(
                rollout_buffer_true_loss_no_value_bootstrap,
                agent.gamma,
                get_episode_length(self.env_factory()),
            )
            loss_checkpoint = evaluate_agent_losses(agent, rollout_buffer_true_loss)

            cliff_test_parameters = [
                p + self.cliff_test_distance * g
                for p, g in zip(agent.policy.parameters(), normalized_gradient)
            ]
            cliff_test_agent_spec = self.agent_spec.copy_with_new_weights(
                cliff_test_parameters
            )
            reward_cliff_test = evaluate_agent_returns(
                cliff_test_agent_spec,
                self.env_factory,
                self.num_steps_reward_eval,
            )
            loss_cliff_test = evaluate_agent_losses(
                cliff_test_agent_spec, rollout_buffer_true_loss
            )

            checkpoint_is_cliff = self.cliff_criterion(
                reward_checkpoint, reward_cliff_test
            )

            self.dump_pre_update_results(
                env_step,
                checkpoint_is_cliff,
                reward_checkpoint,
                loss_checkpoint,
                reward_cliff_test,
                loss_cliff_test,
                overwrite_results,
            )

        losses_after_update = []
        rewards_after_update = []
        overrides = [{}] + list(self.algorithm_overrides["ppo"])
        for curr_overrides in overrides:
            if self._overrides_to_str(curr_overrides) not in curr_results.get(
                "ppo", {}
            ):
                agent_spec_overrides = AgentSpec(
                    self.agent_spec.checkpoint_path,
                    self.agent_spec.device,
                    self.agent_spec.override_weights,
                    {**self.agent_spec.agent_kwargs, **curr_overrides},
                )
                agent = agent_spec_overrides.create_agent(self.env_factory())
                # TODO: This should be changed! --> Put it into the config?
                agent.learn(2048)

                rewards_after_update.append(
                    evaluate_agent_returns(
                        agent, self.env_factory, self.num_steps_reward_eval
                    )
                )
                losses_after_update.append(
                    evaluate_agent_losses(agent, rollout_buffer_true_loss)
                )

        self.dump_update_results(
            env_step, rewards_after_update, losses_after_update, "ppo", overrides
        )

        # TODO: Should probably also compute the performance gain for ground truth gradient steps
        # TODO: Dump agent config somewhere
        # TODO: Should be parallelized (at least put stuff on a separate process to avoid clogging the main process)

        # Dict env_step (maybe also store the parameters of the analysis somewhere?) -->
        # cliff result, performance drop, reward_checkpoint, reward_cliff_test (discounted, undiscounted), losses
        # Maybe already add support for logging stuff like the learning rate and number of steps (since that is varied in the paper)
        #   --> The paper also does not use 32 steps, but rather 128 and 2048 (128 due to hyperparameters?)
        # If we already know that this location is a cliff, we do not need to check it again

        return TensorboardLogs()

    def cliff_criterion(
        self,
        reward_checkpoint: ReturnEvaluationResult,
        reward_cliff_test: ReturnEvaluationResult,
    ) -> bool:
        reward_checkpoint = np.mean(reward_checkpoint.rewards_undiscounted)
        reward_cliff_test = np.mean(reward_cliff_test.rewards_undiscounted)
        return (
            reward_cliff_test <= self.cliff_reward_decrease * reward_checkpoint
            and (reward_cliff_test - reward_checkpoint) / self.global_reward_range
            > self.cliff_reward_decrease_global
        )

    def dump_pre_update_results(
        self,
        env_step: int,
        checkpoint_is_cliff: bool,
        reward_checkpoint: ReturnEvaluationResult,
        loss_checkpoint: LossEvaluationResult,
        reward_cliff_test: ReturnEvaluationResult,
        loss_cliff_test: LossEvaluationResult,
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
                "checkpoint_is_cliff" not in curr_results
                and "reward_checkpoint" not in curr_results
                and "loss_checkpoint" not in curr_results
            ), (
                f"Override not set but checkpoint results are non-empty "
                f"(step: {env_step}, results file: {self.results_file})."
            )

            curr_results["checkpoint_is_cliff"] = checkpoint_is_cliff
            curr_results["reward_checkpoint"] = {
                k: np.mean(v).item()
                for k, v in dataclasses.asdict(reward_checkpoint).items()
            }
            curr_results["loss_checkpoint"] = {
                k: np.mean(v).item()
                for k, v in dataclasses.asdict(loss_checkpoint).items()
            }
            curr_results["reward_cliff_test"] = {
                k: np.mean(v).item()
                for k, v in dataclasses.asdict(reward_cliff_test).items()
            }
            curr_results["loss_cliff_test"] = {
                k: np.mean(v).item()
                for k, v in dataclasses.asdict(loss_cliff_test).items()
            }

    def dump_update_results(
        self,
        env_step: int,
        rewards_after_update: Sequence[ReturnEvaluationResult],
        losses_after_update: Sequence[LossEvaluationResult],
        algorithm_name: str,
        hyperparameter_overrides: Sequence[Dict[str, Any]],
    ) -> None:
        lock = filelock.FileLock(
            self.results_file.with_suffix(self.results_file.suffix + ".lock")
        )
        with lock.acquire(timeout=60):
            results = self._read_results_file()
            if env_step not in results:
                results[env_step] = {}

            for (overrides, rew_after, loss_after) in zip(
                hyperparameter_overrides,
                rewards_after_update,
                losses_after_update,
            ):
                curr_results = results[env_step]
                if algorithm_name not in curr_results:
                    curr_results[algorithm_name] = {}
                curr_results = curr_results[algorithm_name]
                overrides_str = self._overrides_to_str(overrides)
                if overrides_str not in curr_results:
                    curr_results[overrides_str] = {}
                curr_results = curr_results[overrides_str]
                curr_results["reward_update"] = {
                    k: np.mean(v).item()
                    for k, v in dataclasses.asdict(rew_after).items()
                }
                curr_results["loss_update"] = {
                    k: np.mean(v).item()
                    for k, v in dataclasses.asdict(loss_after).items()
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
        curr_results = results[env_step]
        return (
            "checkpoint_is_cliff" in curr_results
            and "reward_checkpoint" in curr_results
            and "loss_checkpoint" in curr_results
            and "loss_cliff_test" in curr_results
            and "reward_cliff_test" in curr_results
        )
