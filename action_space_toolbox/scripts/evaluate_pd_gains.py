import random
import time
from argparse import ArgumentParser
from typing import List, Sequence

import gym
import gym.envs.mujoco
import numpy as np

from action_space_toolbox import PositionControlWrapper
from action_space_toolbox.control_modes.check_wrapped import check_wrapped
from action_space_toolbox.util.angles import normalize_angle


def sample_targets(env_id: str, num_targets: int) -> List[np.ndarray]:
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
    return random.choices(dof_positions, k=num_targets)


def visualize_targets(env, targets: Sequence[np.ndarray]) -> None:
    for pos in targets:
        env.reset()
        if isinstance(env.unwrapped, gym.envs.mujoco.MujocoEnv):
            full_pos = env.sim.data.qpos
            full_pos[env.actuated_joints] = pos
            env.set_state(full_pos, np.zeros_like(full_pos))
        else:
            env.unwrapped.state = (pos, np.zeros_like(pos))
        env.render()
        time.sleep(1)


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
    parser.add_argument("--p_gains", type=float, nargs="+")
    parser.add_argument("--d_gains", type=float, nargs="+")
    parser.add_argument("--num-targets", type=int, default=10)
    parser.add_argument("--visualize-targets", action="store_true")
    args = parser.parse_args()

    targets = sample_targets(args.env_id, args.num_targets)
    env = gym.make(args.env_id, p_gains=args.p_gains, d_gains=args.d_gains)
    if args.visualize_targets:
        visualize_targets(env, targets)
    loss = evaluate_pd_gains(env, targets, render=True)
    print(f"Average joint error: {loss}")
