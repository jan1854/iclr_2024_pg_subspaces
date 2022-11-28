import stable_baselines3.common.callbacks


class FixEpInfoBufferCallback(stable_baselines3.common.callbacks.BaseCallback):
    """
    stable-baselines3 always averages the performance of the agent over the last 100 episodes. This data often includes
    rollouts collected with a policy from many updates ago (e.g., in the case of the default PPO configuration on
    Cartpole-v0: more than 10 updates). This callback fixes this issue by clearing the ep_info_buffer before new
    rollouts are collected.
    """

    def __init__(self):
        super().__init__(verbose=0)

    def _on_rollout_start(self) -> None:
        algorithm = self.locals["self"]
        algorithm.ep_info_buffer.clear()

    def _on_step(self) -> bool:
        return True
