import functools
import logging
import pickle
from pathlib import Path
from typing import List, Tuple, Callable, Union

import gym
import numpy as np
import stable_baselines3.common.callbacks
import torch
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.type_aliases import RolloutBufferSamples
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv

from action_space_toolbox.analysis.analysis import Analysis
from action_space_toolbox.analysis.gradient_analysis.value_function import (
    ValueFunctionTrainer,
    ValueFunction,
)
from action_space_toolbox.util import metrics
from action_space_toolbox.util.get_episode_length import get_episode_length
from action_space_toolbox.util.sb3_training import ppo_loss, fill_rollout_buffer
from action_space_toolbox.util.tensorboard_logs import TensorboardLogs
from action_space_toolbox.util.tensors import weighted_mean

logger = logging.getLogger(__name__)


# TODO: This class is quite a mess, split up the functionality into multiple classes, and reduce the number of arguments
#   passed around everywhere
class GradientAnalysis(Analysis):
    def __init__(
        self,
        env_factory: Callable[[], gym.Env],
        agent_factory: Callable[[Union[gym.Env, VecEnv]], stable_baselines3.ppo.PPO],
        run_dir: Path,
        num_gradient_estimates: int = 500,
        samples_true_gradient: int = 10**7,
        gt_value_function_analysis: bool = False,
        epochs_gt_value_function_training: int = 1,
        dump_gt_value_function_dataset: bool = False,
    ):
        super().__init__(
            "gradient_analysis", env_factory, agent_factory, run_dir, num_processes=1
        )
        self.num_gradient_estimates = num_gradient_estimates
        self.samples_true_gradient = samples_true_gradient
        self.gt_value_function_analysis = gt_value_function_analysis
        self.epochs_gt_value_function_training = epochs_gt_value_function_training
        self._dump_gt_value_function_dataset = dump_gt_value_function_dataset

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
            agent.batch_size * self.num_gradient_estimates,
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
        gradient_similarity_true = self.compute_similarity_true_gradient(
            agent, rollout_buffer_gradient_estimates, rollout_buffer_true_gradient
        )
        gradient_similarity_estimates = self.compute_gradient_estimates_similarity(
            agent, env, rollout_buffer_gradient_estimates
        )
        # Measure how well the value function predicts the "true" value
        value_function_gae_mre = self.compute_value_function_gae_mre(
            agent, rollout_buffer_true_gradient
        )

        logs = TensorboardLogs()
        logs.add_scalar(
            f"gradient_analysis/{self.analysis_run_id}/similarity_estimates_true_gradient",
            gradient_similarity_true,
            env_step,
        )
        logs.add_scalar(
            f"gradient_analysis/{self.analysis_run_id}/similarity_gradient_estimates",
            gradient_similarity_estimates,
            env_step,
        )
        logs.add_scalar(
            f"gradient_analysis/{self.analysis_run_id}/value_function_gae_mre",
            value_function_gae_mre,
            env_step,
        )

        if self.gt_value_function_analysis:
            states_gt, values_gt = self._compute_gt_values(
                rollout_buffer_true_gradient, get_episode_length(env.envs[0])
            )
            states_gt = torch.tensor(states_gt, device=agent.device)
            values_gt = torch.tensor(values_gt, device=agent.device)
            if self._dump_gt_value_function_dataset:
                gt_value_function_dataset_dir = Path("gt_value_function_datasets")
                gt_value_function_dataset_dir.mkdir(exist_ok=True)
                dump_path = gt_value_function_dataset_dir / f"step_{env_step:07d}.pkl"
                with dump_path.open("wb") as file:
                    pickle.dump({"states": states_gt, "values": values_gt}, file)
            value_function_gt_mre = metrics.mean_relative_error(
                policy.predict_values(states_gt), values_gt
            )

            gt_value_function = ValueFunction(
                policy.features_dim,
                policy.net_arch,
                policy.activation_fn,
                policy.ortho_init,
                policy.init_weights,
                policy.device,
            )
            # TODO: stable-baselines3 supports learning rate schedules, this could make using simply the current learning
            #  rate sub-optimal
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
            rollout_buffer_gradient_estimates_gt_values = (
                self._create_and_fill_gt_rollout_buffer(
                    agent, rollout_buffer_gradient_estimates, gt_value_function
                )
            )
            gradient_similarity_true_gt_value = self.compute_similarity_true_gradient(
                agent,
                rollout_buffer_gradient_estimates_gt_values,
                rollout_buffer_true_gradient,
            )
            gradient_similarity_estimates_gt_value = (
                self.compute_gradient_estimates_similarity(
                    agent, env, rollout_buffer_gradient_estimates_gt_values
                )
            )

            logs.add_scalar(
                f"gradient_analysis/{self.analysis_run_id}/value_function_gt_mre",
                value_function_gt_mre,
                env_step,
            )
            logs.add_scalar(
                f"gradient_analysis/{self.analysis_run_id}/similarity_estimates_true_gradient_gt_value_function",
                gradient_similarity_true_gt_value,
                env_step,
            )
            logs.add_scalar(
                f"gradient_analysis/{self.analysis_run_id}/similarity_gradient_estimates_gt_value_function",
                gradient_similarity_estimates_gt_value,
                env_step,
            )

            layout = {
                f"gradient_analysis/{self.analysis_run_id}/": {
                    "similarity_estimates_true_gradient": [
                        "Multiline",
                        [
                            f"gradient_analysis/{self.analysis_run_id}/similarity_estimates_true_gradient",
                            f"gradient_analysis/{self.analysis_run_id}/similarity_estimates_true_gradient_gt_value_function",
                        ],
                    ],
                    f"similarity_gradient_estimates/{self.analysis_run_id}/": [
                        "Multiline",
                        [
                            f"gradient_analysis/{self.analysis_run_id}/similarity_gradient_estimates",
                            f"gradient_analysis/{self.analysis_run_id}/similarity_gradient_estimates_gt_value_function",
                        ],
                    ],
                }
            }
            logs.add_custom_scalars(layout)
        return logs

    def compute_similarity_true_gradient(
        self,
        agent: stable_baselines3.ppo.PPO,
        rollout_buffer: RolloutBuffer,
        rollout_buffer_true_gradient: RolloutBuffer,
    ) -> float:

        # Compute the gradients in batches to avoid running out of memory on the GPU
        batches = list(rollout_buffer_true_gradient.get(100_000))
        gradients_batches = [self._ppo_gradient(agent, batch) for batch in batches]
        true_gradient = weighted_mean(
            gradients_batches, [len(batch.observations) for batch in batches]
        )

        estimated_gradients = torch.stack(
            [
                self._ppo_gradient(agent, rollout_data)
                for rollout_data in rollout_buffer.get(agent.batch_size)
            ]
        )
        cos_similarities = self._cosine_similarities(
            true_gradient.unsqueeze(0), estimated_gradients
        )
        return cos_similarities.mean().item()

    def compute_gradient_estimates_similarity(
        self,
        agent: stable_baselines3.ppo.PPO,
        env: VecEnv,
        rollout_buffer: RolloutBuffer,
    ) -> float:
        assert isinstance(agent, stable_baselines3.ppo.PPO)
        env.reset()
        gradients = torch.stack(
            [
                self._ppo_gradient(agent, rollout_data)
                for rollout_data in rollout_buffer.get(agent.batch_size)
            ]
        )
        dist_matrix = self._cosine_similarities(gradients, gradients)

        # Calculate the average pairwise similarity
        dist_sum = (dist_matrix.sum() - torch.trace(dist_matrix)) / 2
        n = dist_matrix.shape[0]
        return (dist_sum / (n * (n - 1) / 2)).item()

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

    def _create_and_fill_gt_rollout_buffer(
        self,
        agent: stable_baselines3.ppo.PPO,
        rollout_buffer_gradient_estimates: RolloutBuffer,
        value_function: ValueFunction,
    ) -> RolloutBuffer:
        # stable-baselines3 assumes that the rollout buffer is full. Since the exact number of samples depends on the
        # number of episodes that are terminated compared to those that are truncated, we cannot pre-allocate the buffer
        # and instead need to create an adequately-sized buffer for the data at hand.
        episode_length = get_episode_length(agent.env.envs[0])

        (
            states_episodes,
            actions_episodes,
            rewards_episodes,
            log_probs_episodes,
        ) = self._rollout_buffer_split_episodes(rollout_buffer_gradient_estimates)

        data = []
        for states_ep, actions_ep, rewards_ep, log_probs_ep in zip(
            states_episodes, actions_episodes, rewards_episodes, log_probs_episodes
        ):
            episode_truncated = len(states_ep) == episode_length
            if episode_truncated:
                states_value = states_ep[: len(states_ep) // 2]
                actions_value = actions_ep[: len(actions_ep) // 2]
                rewards_value = rewards_ep[: len(rewards_ep) // 2]
                log_probs_value = log_probs_ep[: len(log_probs_ep) // 2]
                # Handle episode truncation by adding the value of the terminal state (as in stable_baselines3's
                # OnPolicyAlgorithm.collect_rollouts()). We use the value function learned during RL training here
                # since the analysis is only about what happens if we use the ground truth value function
                # **as baseline**.
                with torch.no_grad():  # TODO: This should probably be batched
                    terminal_value = (
                        agent.policy.predict_values(
                            torch.tensor(
                                states_ep[len(states_ep) // 2],
                                device=agent.policy.device,
                            ).unsqueeze(0)
                        )[0]
                        .cpu()
                        .numpy()
                    ).squeeze(0)
                rewards_value[-1] += agent.gamma * terminal_value
            else:
                states_value = states_ep
                actions_value = actions_ep
                rewards_value = rewards_ep
                log_probs_value = log_probs_ep
            episode_start = True
            for state, action, reward, log_prob in zip(
                states_value, actions_value, rewards_value, log_probs_value
            ):
                with torch.no_grad():
                    value = value_function(torch.tensor(state, device=agent.device))
                data.append(
                    (
                        state,
                        action,
                        reward,
                        np.array([episode_start]),
                        value,
                        torch.tensor(log_prob),
                    )
                )
                episode_start = False

        assert isinstance(agent, stable_baselines3.ppo.PPO)
        rollout_buffer = RolloutBuffer(
            len(data),
            agent.observation_space,
            agent.action_space,
            agent.device,
            agent.gae_lambda,
            agent.gamma,
        )
        for d in data:
            rollout_buffer.add(*d)
        return rollout_buffer

    def _rollout_buffer_split_episodes(
        self, rollout_buffer: RolloutBuffer
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        states_episodes = []
        actions_episodes = []
        rewards_episodes = []
        log_probs_episodes = []
        start_idx = 0
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

    @staticmethod
    def _cosine_similarities(
        t1: torch.Tensor, t2: torch.Tensor, eps: float = 1e-8
    ) -> torch.Tensor:
        t1_norm = t1.norm(dim=1)[:, None]
        t2_norm = t2.norm(dim=1)[:, None]
        t1_normalized = t1 / torch.clamp(t1_norm, min=eps)
        t2_normalized = t2 / torch.clamp(t2_norm, min=eps)
        sim_mt = torch.mm(t1_normalized, t2_normalized.transpose(0, 1))
        return sim_mt

    def _ppo_gradient(
        self, agent: stable_baselines3.ppo.PPO, rollout_data: RolloutBufferSamples
    ) -> torch.Tensor:
        loss = ppo_loss(agent, rollout_data)
        agent.policy.zero_grad()
        loss.backward()
        return torch.cat([p.grad.flatten() for p in agent.policy.parameters()])
