import random
import time
from argparse import ArgumentParser
from typing import Tuple, List, Sequence

import gym
import numpy as np

from action_space_toolbox import PositionControlWrapper
from action_space_toolbox.control_modes.check_wrapped import check_wrapped


def determine_target_dof_positions(env_id) -> List[np.ndarray]:
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
        # Leave out the first states since these can be hard to reach for certain initializations (e.g. the pendulum
        # might be initialized upright)
        dof_positions.extend(dof_positions_curr_episode[10:])
    return random.choices(dof_positions, k=10)


def visualize_target_positions(env, target_dof_positions: Sequence[np.ndarray]) -> None:
    for pos in target_dof_positions:
        full_pos = env.sim.data.qpos
        full_pos[env.actuated_joints] = pos
        env.set_state(full_pos, np.zeros_like(full_pos))
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
        loss = evaluate_pd_gains(env, dof_positions)
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
    num_episodes: int = 10,
    render=False,
) -> float:
    assert check_wrapped(env, PositionControlWrapper)
    joint_errors = []
    for target_dof_position in target_dof_positions:
        for _ in range(num_episodes):
            done = False
            env.reset()
            joint_errors.append(0.0)
            while not done:
                _, _, done, _ = env.step(target_dof_position)
                if render:
                    env.render()
                joint_errors[-1] += np.mean(
                    np.abs(env.dof_positions - target_dof_position)
                )

    return np.mean(joint_errors)  # type: ignore


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("env_id", type=str)
    parser.add_argument("num_iterations", type=int)
    parser.add_argument("p_gains_exp_low", type=float)
    parser.add_argument("p_gains_exp_high", type=float)
    parser.add_argument("d_gains_exp_low", type=float)
    parser.add_argument("d_gains_exp_high", type=float)
    args = parser.parse_args()

    random.seed(42)
    target_dof_positions = determine_target_dof_positions(args.env_id)

    tuned_gains = tune_pd_gains(
        args.env_id,
        target_dof_positions,
        args.num_iterations,
        args.p_gains_exp_low,
        args.p_gains_exp_high,
        args.d_gains_exp_low,
        args.d_gains_exp_high,
    )

    env = gym.make(args.env_id, p_gains=tuned_gains[0], d_gains=tuned_gains[1])
    input("Press any key to visualize the optimized controllers.")
    evaluate_pd_gains(env, target_dof_positions, num_episodes=1, render=True)
