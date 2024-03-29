import math

import stable_baselines3.common.callbacks
from stable_baselines3.common.utils import safe_mean


class AdditionalTrainingMetricsCallback(
    stable_baselines3.common.callbacks.BaseCallback
):
    """
    stable-baselines3 always averages the performance of the agent over the last 100 episodes. This data often includes
    rollouts collected with a policy from many updates ago (e.g., in the case of the default PPO configuration on
    Cartpole-v0: more than 10 updates). This callback fixes this issue by clearing the ep_info_buffer before new
    rollouts are collected.
    """

    def __init__(self):
        super().__init__(verbose=0)

    def _on_rollout_end(self) -> None:
        algorithm = self.locals["self"]
        if len(algorithm.ep_info_buffer) > 0:
            # Log the current values since we are logging later with the number of gradient steps instead of the number
            # of environment steps (otherwise all metrics would be logged with the number of gradient steps).
            self.logger.dump(algorithm.num_timesteps)
            self.logger.record(
                "rollout/ep_rew_mean_gradient_steps",
                safe_mean([ep_info["r"] for ep_info in algorithm.ep_info_buffer]),
            )
            timesteps_per_policy_update = algorithm.n_steps * algorithm.n_envs
            curr_num_policy_updates = (
                algorithm.num_timesteps // timesteps_per_policy_update
            )
            if isinstance(algorithm, stable_baselines3.a2c.A2C):
                gradient_steps_per_epoch = 1
                n_epochs = 1
            else:
                # The last gradient step is done on a smaller batch if the number of samples is not divisible by the
                # batch size
                gradient_steps_per_epoch = math.ceil(
                    timesteps_per_policy_update / algorithm.batch_size
                )
                n_epochs = algorithm.n_epochs
            curr_num_gradient_steps = (
                curr_num_policy_updates * n_epochs * gradient_steps_per_epoch
            )
            self.logger.dump(step=curr_num_gradient_steps)

    def _on_step(self) -> bool:
        return True
