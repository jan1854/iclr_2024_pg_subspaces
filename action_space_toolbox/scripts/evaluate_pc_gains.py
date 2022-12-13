import json
import random
import time
from argparse import ArgumentParser
from pathlib import Path
from typing import List, Sequence, Optional

import dm_control.suite.cheetah
import dmc2gym.wrappers
import gym
import gym.envs.mujoco
import numpy as np
from dm_control import mjcf

from action_space_toolbox import FixedGainsPositionControlWrapper
from action_space_toolbox.control_modes.check_wrapped import check_wrapped
from action_space_toolbox.scripts.render import Renderer
from action_space_toolbox.util.angles import normalize_angle


def prepare_env(env: gym.Env) -> None:
    if env.spec.id == "Pendulum_PC-v1" or env.spec.id == "Reacher_PC-v2":
        pass
    elif env.spec.id == "dmc_Cheetah-run_PC-v1":
        model, assets = dm_control.suite.cheetah.get_model_and_assets()
        model = mjcf.from_xml_string(model, assets=assets)
        model.find("body", "torso").pos = np.array([0.0, 0.0, 0.9])
        # Increase stiffness and damping for the root joints to make them quasi static
        for joint_name in ["rootx", "rootz", "rooty"]:
            joint = model.find("joint", joint_name)
            joint.stiffness = 10000000.0
            joint.damping = 10000000.0

        physics = dm_control.suite.cheetah.Physics.from_xml_string(
            model.to_xml_string()
        )
        env.unwrapped._env._physics = physics
    else:
        raise ValueError(f"Unsupported environment: {env.unwrapped.spec.id}")


def sample_targets(env_id: str, num_targets: int) -> List[np.ndarray]:
    # Unwrap the env to get back the original action space
    env = gym.make(env_id, normalize=False)
    env.seed(42)
    prepare_env(env)
    tmp_env = env
    while not isinstance(tmp_env.env, FixedGainsPositionControlWrapper):
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
    renderer = Renderer()
    for pos in targets:
        env.reset()
        if isinstance(env.unwrapped, gym.envs.mujoco.MujocoEnv):
            full_pos = env.sim.data.qpos
            full_pos[env.actuated_joints] = pos
            env.set_state(full_pos, np.zeros_like(full_pos))
        elif isinstance(env.unwrapped, dmc2gym.wrappers.DMCWrapper):
            full_pos = env.physics.data.qpos
            full_pos[env.actuated_joints] = pos
            env.physics.set_state(np.concatenate((full_pos, np.zeros_like(full_pos))))
            env.step(np.zeros(env.action_space.shape))
        else:
            env.unwrapped.state = (pos, np.zeros_like(pos))
        renderer.render_frame(env)
        time.sleep(1)


def evaluate_pc_gains(
    env,
    target_actuator_positions: Sequence[np.ndarray],
    repetitions_per_target: int = 1,
    render: bool = False,
    max_steps_per_episode: Optional[int] = None,
) -> float:
    assert check_wrapped(env, FixedGainsPositionControlWrapper)
    joint_errors = []
    renderer = Renderer()
    for target_actuator_position in target_actuator_positions:
        for repetition in range(repetitions_per_target):
            done = False
            env.reset()
            joint_errors.append(0.0)
            i = 0
            while not done:
                _, _, done, _ = env.step(target_actuator_position)
                if render:
                    renderer.render_frame(env)
                diff = env.actuator_positions - target_actuator_position
                joint_diff = diff * ~np.array(env.actuators_revolute) + normalize_angle(
                    diff
                ) * np.array(env.actuators_revolute)
                joint_errors[-1] += np.mean(np.abs(joint_diff))
                i += 1
                if max_steps_per_episode is not None and i >= max_steps_per_episode:
                    break
            joint_errors[-1] /= i

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
    p_gains = args.p_gains[0] if len(args.p_gains) == 1 else args.p_gains
    d_gains = args.d_gains[0] if len(args.d_gains) == 1 else args.d_gains
    env = gym.make("dmc_Cheetah-run_PC-v1")
    prepare_env(env)
    env.reset()
    env = gym.make(args.env_id, p_gains=p_gains, d_gains=d_gains, normalize=False)
    prepare_env(env)
    if args.visualize_targets:
        visualize_targets(env, targets)
    loss = evaluate_pc_gains(env, targets, render=True)
    print(f"Average joint error: {loss}")
