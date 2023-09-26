import pickle
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt


class ConvergenceCriterion:
    def __init__(
        self,
        smoothing_steps: int,
        training_threshold: float,
        convergence_threshold: float,
    ):
        self.smoothing_timesteps = smoothing_steps
        self.training_threshold = training_threshold
        self.convergence_threshold = convergence_threshold

    def split_steps(self, timesteps, rewards):
        timesteps = np.asarray(timesteps)
        rewards = np.asarray(rewards)
        window_steps = round(self.smoothing_timesteps / (timesteps[1] - timesteps[0]))
        smoothed_rewards = np.array(
            [
                np.mean(rewards[max(i - window_steps, 0) : i + window_steps])
                for i in range(len(rewards))
            ]
        )
        converged_reward = max(smoothed_rewards)
        initial_reward = smoothed_rewards[0]
        rel_improvement = (smoothed_rewards - initial_reward) / (
            converged_reward - initial_reward
        )
        idx_training_begin = None
        for i, impr in enumerate(rel_improvement):
            if impr > self.training_threshold:
                idx_training_begin = i
                break

        idx_convergence_begin = None
        max_performance_reached = False
        for i, impr in reversed(list(enumerate(rel_improvement))):
            if not max_performance_reached and impr == 1.0:
                max_performance_reached = True
            if max_performance_reached and impr < self.convergence_threshold:
                idx_convergence_begin = i
                break

        return (
            (timesteps[:idx_training_begin], rewards[:idx_training_begin]),
            (
                timesteps[idx_training_begin:idx_convergence_begin],
                rewards[idx_training_begin:idx_convergence_begin],
            ),
            (timesteps[idx_convergence_begin:], rewards[idx_convergence_begin:]),
        )


if __name__ == "__main__":
    criterion = ConvergenceCriterion(20000, 0.1, 0.9)
    vis_dir = Path(
        "/home/jschneider/Seafile/PhD/project_optimal_action_spaces/iclr_2024/vis"
    )
    for cache_file in Path(
        "/home/jschneider/Seafile/PhD/project_optimal_action_spaces/iclr_2024/learning_curve_cache"
    ).iterdir():
        with cache_file.open("rb") as f:
            res = pickle.load(f)
            res = {k: np.array(v) for k, v in res.items()}
        for i in range(10):
            init, train, conv = criterion.split_steps(res["steps"], res["reward"][:, i])
            plt.plot(res["steps"], res["reward"][:, i])
            plt.axvline(x=init[0][-1], color="black", linestyle="--")
            plt.axvline(x=conv[0][0], color="black", linestyle="--")
            plt.savefig(vis_dir / f"{cache_file.with_suffix('').name}_{i}.pdf")
            plt.close()
        init, train, conv = criterion.split_steps(
            res["steps"], res["reward"].mean(axis=1)
        )
        plt.plot(res["steps"], res["reward"].mean(axis=1))
        plt.axvline(x=init[0][-1], color="black", linestyle="--")
        plt.axvline(x=conv[0][0], color="black", linestyle="--")
        plt.savefig(vis_dir / f"avg_{cache_file.with_suffix('').name}.pdf")
        plt.close()
