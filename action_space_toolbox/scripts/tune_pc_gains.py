import json
import random
from argparse import ArgumentParser
from pathlib import Path
from typing import Tuple, Sequence

import gym
import gym.envs.mujoco
import numpy as np
from tqdm import trange

from action_space_toolbox.scripts.evaluate_pc_gains import (
    evaluate_pc_gains,
    sample_targets,
    visualize_targets,
)


def tune_pc_gains(
    env_id: str,
    actuator_positions: Sequence[np.ndarray],
    num_iterations: int,
    p_gains_exp_low: float,
    p_gains_exp_high: float,
    d_gains_exp_low: float,
    d_gains_exp_high: float,
    repetitions_per_target: int,
) -> Tuple[np.ndarray, np.ndarray]:
    temp_env = gym.make(env_id)
    action_shape = temp_env.action_space.shape
    best_loss = float("inf")
    best_gains = None
    progress_bar = trange(num_iterations)
    losses = []
    for i in progress_bar:
        p_gains = np.around(
            10
            ** np.random.uniform(p_gains_exp_low, p_gains_exp_high, size=action_shape),
            decimals=2,
        )
        d_gains = np.around(
            10
            ** np.random.uniform(d_gains_exp_low, d_gains_exp_high, size=action_shape),
            decimals=2,
        )
        env = gym.make(env_id, p_gains=p_gains, d_gains=d_gains, normalize=False)
        env.seed(42)
        loss = evaluate_pc_gains(
            env, actuator_positions, repetitions_per_target, max_steps_per_episode=200
        )
        losses.append(loss)
        if loss < best_loss:
            best_gains = (p_gains, d_gains)
            best_loss = loss
        progress_bar.set_description(
            f"Tuning gains. Best gains: p: {best_gains[0]}, d: {best_gains[1]}, "
            f"loss: {best_loss:.4f}, average_loss: {np.mean(losses):.4f}."
        )

    return best_gains


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("env_id", type=str)
    parser.add_argument("num_iterations", type=int)
    parser.add_argument("p_gains_exp_low", type=float)
    parser.add_argument("p_gains_exp_high", type=float)
    parser.add_argument("d_gains_exp_low", type=float)
    parser.add_argument("d_gains_exp_high", type=float)
    parser.add_argument("--visualize-targets", action="store_true")
    parser.add_argument("--targets-to-sample", type=int, default=25)
    parser.add_argument("--repetitions-per-target", type=int, default=1)
    args = parser.parse_args()

    random.seed(42)
    fixed_targets_path = Path(__file__).parent / "res" / "pc_tuning_fixed_targets.json"
    with fixed_targets_path.open("r") as fixed_targets_file:
        fixed_targets = json.load(fixed_targets_file)
    if args.env_id in fixed_targets:
        target_actuator_positions = np.array(fixed_targets[args.env_id])
    else:
        target_actuator_positions = sample_targets(args.env_id, args.targets_to_sample)

    if args.visualize_targets:
        env = gym.make(args.env_id)
        visualize_targets(env, target_actuator_positions)

    tuned_gains = tune_pc_gains(
        args.env_id,
        target_actuator_positions,
        args.num_iterations,
        args.p_gains_exp_low,
        args.p_gains_exp_high,
        args.d_gains_exp_low,
        args.d_gains_exp_high,
        args.repetitions_per_target,
    )

    env = gym.make(
        args.env_id, p_gains=tuned_gains[0], d_gains=tuned_gains[1], normalize=False
    )
    input("Press any key to visualize the optimized controllers.")
    evaluate_pc_gains(
        env, target_actuator_positions, repetitions_per_target=1, render=True
    )
