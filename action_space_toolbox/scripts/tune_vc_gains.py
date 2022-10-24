import random
from argparse import ArgumentParser
from typing import Tuple, Sequence

import gym
import gym.envs.mujoco
import numpy as np

from action_space_toolbox.scripts.evaluate_vc_gains import (
    evaluate_vc_gains,
    sample_targets,
    State,
    visualize_targets,
)


def tune_vc_gains(
    env_id: str,
    targets: Sequence[Tuple[State, np.ndarray]],
    num_iterations: int,
    gains_exp_low: float,
    gains_exp_high: float,
    repetitions_per_target: int,
) -> Tuple[np.ndarray, np.ndarray]:
    temp_env = gym.make(env_id)
    action_shape = temp_env.action_space.shape
    best_loss = float("inf")
    best_gains = None
    for i in range(num_iterations):
        gains = 10 ** np.random.uniform(
            gains_exp_low, gains_exp_high, size=action_shape
        )
        env = gym.make(env_id, gains=gains)
        env.seed(42)
        loss = evaluate_vc_gains(env, targets, repetitions_per_target)
        print(f"Iteration: {i + 1}/{num_iterations}, gains: {gains}, loss: {loss}")
        if loss < best_loss:
            best_gains = gains
            best_loss = loss

    print(
        f"best: p_gains: {best_gains[0]}, d_gains: {best_gains[1]}, loss: {best_loss}"
    )
    return best_gains


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("env_id", type=str)
    parser.add_argument("num_iterations", type=int)
    parser.add_argument("gains_exp_low", type=float)
    parser.add_argument("gains_exp_high", type=float)
    parser.add_argument("--visualize-targets", action="store_true")
    parser.add_argument("--targets-to-sample", type=int, default=25)
    parser.add_argument("--repetitions-per-target", type=int, default=1)
    args = parser.parse_args()

    random.seed(42)
    targets = sample_targets(args.env_id, args.targets_to_sample)

    if args.visualize_targets:
        env = gym.make(args.env_id)
        visualize_targets(env, targets)

    tuned_gains = tune_vc_gains(
        args.env_id,
        targets,
        args.num_iterations,
        args.gains_exp_low,
        args.gains_exp_high,
        args.repetitions_per_target,
    )

    env = gym.make(args.env_id, gains=tuned_gains)
    input("Press any key to visualize the optimized controllers.")
    evaluate_vc_gains(env, targets, repetitions_per_target=1, render=True)
