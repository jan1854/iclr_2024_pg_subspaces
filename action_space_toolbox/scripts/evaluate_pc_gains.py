import json
import random
import time
from argparse import ArgumentParser
from pathlib import Path
from typing import List, Sequence, Optional

import gym
import gym.envs.mujoco
import numpy as np

from action_space_toolbox import FixedGainPositionControlWrapper
from action_space_toolbox.control_modes.check_wrapped import check_wrapped
from action_space_toolbox.util.angles import normalize_angle


def sample_targets(env_id: str, num_targets: int) -> List[np.ndarray]:
    # Unwrap the env to get back the original action space
    env = gym.make(env_id)
    env.seed(42)
    tmp_env = env
    while not isinstance(tmp_env.env, FixedGainPositionControlWrapper):
        tmp_env = tmp_env.env
    tmp_env.env = tmp_env.env.env
    actuator_positions = []
    for _ in range(5):
        env.reset()
        done = False
        actuator_positions_curr_episode = []
        while not done:
            action = env.action_space.sample()
            _, _, done, _ = env.step(action)
            actuator_positions_curr_episode.append(env.actuator_positions)
        actuator_positions.extend(actuator_positions_curr_episode)
    return random.choices(actuator_positions, k=num_targets)


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


def evaluate_pc_gains(
    env,
    target_actuator_positions: Sequence[np.ndarray],
    repetitions_per_target: int = 1,
    render: bool = False,
    max_steps_per_episode: Optional[int] = None,
) -> float:
    assert check_wrapped(env, FixedGainPositionControlWrapper)
    joint_errors = []
    for target_actuator_position in target_actuator_positions:
        for _ in range(repetitions_per_target):
            done = False
            env.reset()
            joint_errors.append(0.0)
            i = 0
            while not done:
                _, _, done, _ = env.step(target_actuator_position)
                if render:
                    env.render()
                diff = env.actuator_positions - target_actuator_position
                joint_diff = diff * ~np.array(env.actuators_revolute) + normalize_angle(
                    diff
                ) * np.array(env.actuators_revolute)
                joint_errors[-1] += np.mean(np.abs(joint_diff))
                i += 1
                if max_steps_per_episode is not None and i >= max_steps_per_episode:
                    break

    return np.mean(joint_errors)  # type: ignore


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("env_id", type=str)
    parser.add_argument("--p_gains", type=float, nargs="+")
    parser.add_argument("--d_gains", type=float, nargs="+")
    parser.add_argument("--num-targets", type=int, default=10)
    parser.add_argument("--visualize-targets", action="store_true")
    args = parser.parse_args()

    fixed_targets_path = Path(__file__).parent / "res" / "pc_tuning_fixed_targets.json"
    with fixed_targets_path.open("r") as fixed_targets_file:
        fixed_targets = json.load(fixed_targets_file)
    if args.env_id in fixed_targets:
        targets = np.array(fixed_targets[args.env_id])
    else:
        targets = sample_targets(args.env_id, args.num_targets)
    env = gym.make(args.env_id, p_gains=args.p_gains, d_gains=args.d_gains)
    if args.visualize_targets:
        visualize_targets(env, targets)
    loss = evaluate_pc_gains(env, targets, render=True)
    print(f"Average joint error: {loss}")
