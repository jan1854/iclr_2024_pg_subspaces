import functools
import logging
import pickle
from pathlib import Path
from typing import List, Tuple, Callable, Union, Sequence

import gym
import numpy as np
import stable_baselines3.common.callbacks
import torch
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv

from action_space_toolbox.analysis.analysis import Analysis
from action_space_toolbox.analysis.gradient_analysis.gradient_similarity_analysis import (
    GradientSimilarityAnalysis,
)
from action_space_toolbox.analysis.gradient_analysis.value_function import (
    ValueFunctionTrainer,
    ValueFunction,
)
from action_space_toolbox.util import metrics
from action_space_toolbox.util.get_episode_length import get_episode_length
from action_space_toolbox.util.sb3_training import fill_rollout_buffer
from action_space_toolbox.util.tensorboard_logs import TensorboardLogs

logger = logging.getLogger(__name__)


class GradientAnalysis(Analysis):
    def __init__(
        self,
        analysis_run_id: str,
        env_factory: Callable[[], gym.Env],
        agent_factory: Callable[[Union[gym.Env, VecEnv]], stable_baselines3.ppo.PPO],
        run_dir: Path,
        num_gradient_estimates: int = 500,
        samples_true_gradient: int = 10**7,
        samples_different_gradient_estimates: Sequence[int] = tuple(
            10**k for k in range(1, 7)
        ),
        gt_value_function_analysis: bool = False,
        epochs_gt_value_function_training: int = 1,
        dump_gt_value_function_dataset: bool = False,
    ):
        super().__init__(
            "gradient_analysis",
            analysis_run_id,
            env_factory,
            agent_factory,
            run_dir,
            num_processes=1,
        )
        samples_different_gradient_estimates = np.asarray(
            samples_different_gradient_estimates
        )
        if np.any(
            np.array(samples_different_gradient_estimates > samples_true_gradient / 5)
        ):
            logger.warning(
                f"samples_different_gradient_estimates ({samples_different_gradient_estimates.tolist()}) contains "
                f"elements that are large given the value for samples_true_gradient ({samples_true_gradient}), "
                f"these elements will be removed."
            )
            samples_different_gradient_estimates = samples_different_gradient_estimates[
                samples_different_gradient_estimates <= samples_true_gradient / 5
            ]

        self.num_gradient_estimates = num_gradient_estimates
        self.samples_true_gradient = samples_true_gradient
        self.gt_value_function_analysis = gt_value_function_analysis
        self.epochs_gt_value_function_training = epochs_gt_value_function_training
        self._dump_gt_value_function_dataset = dump_gt_value_function_dataset
        self.gradient_similarity_analysis = GradientSimilarityAnalysis(
            num_gradient_estimates,
            samples_different_gradient_estimates,
            analysis_run_id,
        )

    def _do_analysis(
        self,
        process_pool: torch.multiprocessing.Pool,
        env_step: int,
        overwrite_results: bool,
        show_progress: bool,
    ) -> TensorboardLogs:
        return process_pool.apply(
            functools.partial(self.analysis_worker, env_step, show_progress)
        )

    def analysis_worker(self, env_step: int, show_progress: bool) -> TensorboardLogs:
        agent = self.agent_factory(self.env_factory())
        env = DummyVecEnv([self.env_factory])
        value_function_trainer = ValueFunctionTrainer(agent.batch_size)

        rollout_buffer_true_gradient = RolloutBuffer(
            self.samples_true_gradient,
            agent.observation_space,
            agent.action_space,
            agent.device,
            agent.gae_lambda,
            agent.gamma,
        )
        rollout_buffer_gradient_estimates = RolloutBuffer(
            self.samples_true_gradient,
            agent.observation_space,
            agent.action_space,
            agent.device,
            agent.gae_lambda,
            agent.gamma,
        )

        policy = agent.policy
        fill_rollout_buffer(
            agent, env, rollout_buffer_true_gradient, show_progress=show_progress
        )
        fill_rollout_buffer(
            agent, env, rollout_buffer_gradient_estimates, show_progress=False
        )
        logs = self.gradient_similarity_analysis.analyze(
            rollout_buffer_true_gradient,
            rollout_buffer_gradient_estimates,
            agent,
            env_step,
        )

        states_gt, values_gt = self._compute_gt_values(
            rollout_buffer_true_gradient, get_episode_length(env)
        )
        states_gt = torch.tensor(states_gt, device=agent.device)
        values_gt = torch.tensor(values_gt, device=agent.device)

        # Measure how well the value function solves the GAE-objective
        value_function_gae_mre = self.compute_value_function_gae_mre(
            agent, rollout_buffer_true_gradient
        )
        # Measure how well the value function predicts the "true" value
        value_function_gt_mre = metrics.mean_relative_error(
            policy.predict_values(states_gt), values_gt
        )

        logs.add_scalar(
            f"gradient_analysis/{self.analysis_run_id}/value_function_gae_mre",
            value_function_gae_mre,
            env_step,
        )
        logs.add_scalar(
            f"gradient_analysis/{self.analysis_run_id}/value_function_gt_mre",
            value_function_gt_mre,
            env_step,
        )

        if self.gt_value_function_analysis:
            if self._dump_gt_value_function_dataset:
                gt_value_function_dataset_dir = Path("gt_value_function_datasets")
                gt_value_function_dataset_dir.mkdir(exist_ok=True)
                dump_path = gt_value_function_dataset_dir / f"step_{env_step:07d}.pkl"
                with dump_path.open("wb") as file:
                    pickle.dump({"states": states_gt, "values": values_gt}, file)

            gt_value_function = ValueFunction(
                policy.features_dim,
                policy.net_arch,
                policy.activation_fn,
                policy.ortho_init,
                policy.init_weights,
                policy.device,
            )
            # TODO: stable-baselines3 supports learning rate schedules, this could make using simply the current
            #  learning rate suboptimal
            value_function_optimizer = policy.optimizer_class(
                gt_value_function.parameters(),
                lr=policy.optimizer.param_groups[0]["lr"],
                **policy.optimizer_kwargs,
            )
            fit_value_function_logs = value_function_trainer.fit_value_function(
                gt_value_function,
                value_function_optimizer,
                states_gt,
                values_gt,
                self.epochs_gt_value_function_training,
                env_step,
                show_progress=show_progress,
            )
            logs.update(fit_value_function_logs)

        return logs

    def compute_value_function_gae_mre(
        self,
        agent: stable_baselines3.ppo.PPO,
        rollout_buffer_true_gradient: RolloutBuffer,
    ) -> float:
        # Measure how well the value function predicts the values estimated with GAE (the criterion for which the value
        # function was trained)
        states_gae = torch.tensor(
            rollout_buffer_true_gradient.observations, device=agent.device
        )
        values_gae = torch.tensor(
            rollout_buffer_true_gradient.returns, device=agent.device
        )
        return metrics.mean_relative_error(
            agent.policy.predict_values(states_gae), values_gae
        )

    def _compute_gt_values(
        self, rollout_buffer_true_gradient: RolloutBuffer, episode_length
    ) -> Tuple[np.ndarray, np.ndarray]:
        short_episodes_warning_issued = False

        states_episodes, _, rewards_episodes, _ = self._rollout_buffer_split_episodes(
            rollout_buffer_true_gradient
        )

        states = []
        values = []
        for i, (states_ep, rewards_ep) in enumerate(
            zip(states_episodes, rewards_episodes)
        ):
            # Check whether the episode was truncated (time limit reached) since we need to handle this case separately.
            # Note that this is not perfect since the episode might terminate in the last step (this would be counted as
            # truncated with this criterion).
            episode_truncated = len(states_ep) == episode_length
            if episode_truncated:
                # The environment is treated as infinite horizon, just truncated for technical reasons. In the infinite
                # horizon setting, the values should take rewards beyond the truncation horizon into account. We deal
                # with this problem by using only the first half of the transitions of the episode (so that the discount
                # reduces the impact of these rewards).
                states_value = states_ep[: len(states_ep) // 2]
                rewards_value = rewards_ep[: len(rewards_ep) // 2]
                rewards_no_value = rewards_ep[len(rewards_ep) // 2 :]

                gamma_truncated = rollout_buffer_true_gradient.gamma ** (
                    len(rewards_no_value) + 1
                )
                if not short_episodes_warning_issued and gamma_truncated >= 0.05:
                    logger.warning(
                        f"Truncated rewards might have a non-negligible influence on the quality of the true "
                        f"value estimate (gamma^(T / 2 + 1) = {gamma_truncated:.2f})."
                    )
                    # Issue the warning only once to avoid spamming the logs.
                    short_episodes_warning_issued = True

                # The last reward of each episode is actually reward + gamma * value (see stable-baselines3's
                # OnPolicyAlgorithm.collect_rollouts()), so we estimate the quality of the value function with an
                # estimate of itself (but the influence of the estimate should be small due to the discount).
                curr_value = np.sum(
                    rollout_buffer_true_gradient.gamma
                    ** np.arange(rewards_no_value.shape[0])
                    * rewards_no_value[:, 0]
                )
            else:
                curr_value = 0.0
                states_value = states_ep
                rewards_value = rewards_ep
            # We cannot know whether the last episode is shorter because it was terminated or because of the size limit
            # of the rollout buffer, so we just throw it out in this case.
            if i < len(states_episodes) - 1 or len(states_ep) == episode_length:
                values_curr_episode_reversed = []
                for reward in reversed(rewards_value):
                    curr_value = (
                        reward + rollout_buffer_true_gradient.gamma * curr_value
                    )
                    values_curr_episode_reversed.append(curr_value)
                states.extend(states_value)
                values.extend(reversed(values_curr_episode_reversed))
        return np.stack(states, axis=0), np.stack(values, axis=0)

    @classmethod
    def _rollout_buffer_split_episodes(
        cls, rollout_buffer: RolloutBuffer
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        states_episodes = []
        actions_episodes = []
        rewards_episodes = []
        log_probs_episodes = []
        start_idx = 0
        idx = None
        for idx in range(1, rollout_buffer.pos):
            if rollout_buffer.episode_starts[idx]:
                states_episodes.append(rollout_buffer.observations[start_idx:idx])
                actions_episodes.append(rollout_buffer.actions[start_idx:idx])
                rewards_episodes.append(rollout_buffer.rewards[start_idx:idx])
                log_probs_episodes.append(rollout_buffer.log_probs[start_idx:idx])
                start_idx = idx
        states_episodes.append(rollout_buffer.observations[start_idx : idx + 1])
        actions_episodes.append(rollout_buffer.actions[start_idx : idx + 1])
        rewards_episodes.append(rollout_buffer.rewards[start_idx : idx + 1])
        log_probs_episodes.append(rollout_buffer.log_probs[start_idx : idx + 1])
        return states_episodes, actions_episodes, rewards_episodes, log_probs_episodes
