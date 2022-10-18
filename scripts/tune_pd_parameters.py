import json
import random
import time
from argparse import ArgumentParser
from pathlib import Path
from typing import Tuple, List, Sequence

import gym
import gym.envs.mujoco
import numpy as np

from action_space_toolbox import PositionControlWrapper
from action_space_toolbox.control_modes.check_wrapped import check_wrapped
from action_space_toolbox.util.angles import normalize_angle


def determine_target_dof_positions(env_id: str, num_positions: int) -> List[np.ndarray]:
    # Unwrap the env to get back the original action space
    env = gym.make(env_id)
    env.seed(42)
    tmp_env = env
    while not isinstance(tmp_env.env, PositionControlWrapper):
        tmp_env = tmp_env.env
    tmp_env.env = tmp_env.env.env
    dof_positions = []
    for _ in range(5):
        env.reset()
        done = False
        dof_positions_curr_episode = []
        while not done:
            action = env.action_space.sample()
            _, _, done, _ = env.step(action)
            dof_positions_curr_episode.append(env.dof_positions)
        dof_positions.extend(dof_positions_curr_episode)
    return random.choices(dof_positions, k=num_positions)


def visualize_target_positions(env, target_dof_positions: Sequence[np.ndarray]) -> None:
    for pos in target_dof_positions:
        env.reset()
        if isinstance(env.unwrapped, gym.envs.mujoco.MujocoEnv):
            full_pos = env.sim.data.qpos
            full_pos[env.actuated_joints] = pos
            env.set_state(full_pos, np.zeros_like(full_pos))
        else:
            env.unwrapped.state = (pos, np.zeros_like(pos))
        env.render()
        time.sleep(1)


def tune_pd_gains(
    env_id: str,
    dof_positions: Sequence[np.ndarray],
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
    for i in range(num_iterations):
        p_gains = 10 ** np.random.uniform(
            p_gains_exp_low, p_gains_exp_high, size=action_shape
        )
        d_gains = 10 ** np.random.uniform(
            d_gains_exp_low, d_gains_exp_high, size=action_shape
        )
        env = gym.make(env_id, p_gains=p_gains, d_gains=d_gains)
        env.seed(42)
        loss = evaluate_pd_gains(env, dof_positions, repetitions_per_target)
        print(
            f"Iteration: {i + 1}/{num_iterations}, p_gains: {p_gains}, d_gains: {d_gains}, loss: {loss}"
        )
        if loss < best_loss:
            best_gains = (p_gains, d_gains)
            best_loss = loss

    print(
        f"best: p_gains: {best_gains[0]}, d_gains: {best_gains[1]}, loss: {best_loss}"
    )
    return best_gains


def evaluate_pd_gains(
    env,
    target_dof_positions: Sequence[np.ndarray],
    repetitions_per_target: int = 1,
    render: bool = False,
) -> float:
    assert check_wrapped(env, PositionControlWrapper)
    joint_errors = []
    for target_dof_position in target_dof_positions:
        for _ in range(repetitions_per_target):
            done = False
            env.reset()
            joint_errors.append(0.0)
            while not done:
                _, _, done, _ = env.step(target_dof_position)
                if render:
                    env.render()
                diff = env.dof_positions - target_dof_position
                joint_diff = diff * ~np.array(env.dofs_revolute) + normalize_angle(
                    diff
                ) * np.array(env.dofs_revolute)
                joint_errors[-1] += np.mean(np.abs(joint_diff))

    return np.mean(joint_errors)  # type: ignore


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("env_id", type=str)
    parser.add_argument("num_iterations", type=int)
    parser.add_argument("num_joint_positions", type=int)
    parser.add_argument("p_gains_exp_low", type=float)
    parser.add_argument("p_gains_exp_high", type=float)
    parser.add_argument("d_gains_exp_low", type=float)
    parser.add_argument("d_gains_exp_high", type=float)
    parser.add_argument("--visualize-targets", action="store_true")
    parser.add_argument("--repetitions-per-target", type=int, default=1)
    args = parser.parse_args()

    random.seed(42)
    fixed_targets_path = (
        Path(__file__).parent.parent / "res" / "pd_tuning_fixed_targets.json"
    )
    with fixed_targets_path.open("r") as fixed_targets_file:
        fixed_targets = json.load(fixed_targets_file)
    if args.env_id in fixed_targets:
        target_dof_positions = np.array(fixed_targets[args.env_id])
    else:
        target_dof_positions = determine_target_dof_positions(
            args.env_id, args.num_joint_positions
        )

    if args.visualize_targets:
        env = gym.make(args.env_id)
        visualize_target_positions(env, target_dof_positions)

    tuned_gains = tune_pd_gains(
        args.env_id,
        target_dof_positions,
        args.num_iterations,
        args.p_gains_exp_low,
        args.p_gains_exp_high,
        args.d_gains_exp_low,
        args.d_gains_exp_high,
        args.repetitions_per_target,
    )

    env = gym.make(args.env_id, p_gains=tuned_gains[0], d_gains=tuned_gains[1])
    input("Press any key to visualize the optimized controllers.")
    evaluate_pd_gains(env, target_dof_positions, repetitions_per_target=1, render=True)
