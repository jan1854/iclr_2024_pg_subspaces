import random
import time
from argparse import ArgumentParser
from typing import List, Sequence, Tuple

import dmc2gym.wrappers
import gym
import gym.envs.mujoco
import numpy as np

from action_space_toolbox import VelocityControlWrapper
from action_space_toolbox.control_modes.check_wrapped import check_wrapped


State = Tuple[np.ndarray, np.ndarray]


def _set_state(env, state) -> None:
    if isinstance(env.unwrapped, gym.envs.mujoco.MujocoEnv):
        env.set_state(*state)
    elif isinstance(env.unwrapped, dmc2gym.wrappers.DMCWrapper):
        env.physics.set_state(state)
    else:
        env.state = state


def sample_targets(
    env_id: str, num_targets: int, num_episodes: int = 50
) -> List[Tuple[State, np.ndarray]]:
    # Unwrap the env to get back the original action space
    env = gym.make(env_id)
    env.seed(42)
    tmp_env = env
    while not isinstance(tmp_env.env, VelocityControlWrapper):
        tmp_env = tmp_env.env
    tmp_env.env = tmp_env.env.env
    dof_velocities = []
    for _ in range(num_episodes):
        env.reset()
        done = False
        dof_velocities_curr_episode = []
        while not done:
            if isinstance(env.unwrapped, gym.envs.mujoco.MujocoEnv):
                sim_state = env.sim.get_state()
                state = (sim_state.qpos.copy(), sim_state.qvel.copy())
            elif isinstance(env.unwrapped, dmc2gym.wrappers.DMCWrapper):
                state = env.physics.get_state().copy()
            else:
                state = env.state.copy()
            action = env.action_space.sample()
            _, _, done, _ = env.step(action)
            dof_velocities_curr_episode.append((state, env.dof_velocities))
        dof_velocities.extend(dof_velocities_curr_episode)
    return random.choices(dof_velocities, k=num_targets)


def visualize_targets(env, targets: Sequence[Tuple[State, np.ndarray]]) -> None:
    env.reset()
    for target in targets:
        _set_state(env, target[0])
        env.render()
    time.sleep(2)


def evaluate_vc_gains(
    env,
    targets: Sequence[Tuple[State, np.ndarray]],
    repetitions_per_target: int = 1,
    render: bool = False,
) -> float:
    assert check_wrapped(env, VelocityControlWrapper)
    joint_vel_errors = []
    for target in targets:
        for _ in range(repetitions_per_target):
            env.reset()
            _set_state(env, target[0])
            _, _, _, _ = env.step(target[1])
            if render:
                env.render()
            diff = env.dof_velocities - target[1]
            joint_vel_errors.append(np.mean(np.abs(diff)))

    return np.mean(joint_vel_errors)  # type: ignore


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("env_id", type=str)
    parser.add_argument("--gains", type=float, nargs="+")
    parser.add_argument("--num-targets", type=int, default=10)
    parser.add_argument("--visualize-targets", action="store_true")
    args = parser.parse_args()

    targets = sample_targets(args.env_id, args.num_targets)
    env = gym.make(args.env_id, gains=args.gains)
    if args.visualize_targets:
        visualize_targets(env, targets)
    loss = evaluate_vc_gains(env, targets, render=True)
    print(f"Average joint velocity error: {loss}")
