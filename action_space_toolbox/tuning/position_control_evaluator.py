import random
import time
from typing import Optional

import dmc2gym.wrappers
import gym
import gym.envs.mujoco
import numpy as np

from action_space_toolbox.control_modes.position_control.fixed_gains_position_control_wrapper import (
    FixedGainsPositionControlWrapper,
)
from action_space_toolbox.control_modes.check_wrapped import check_wrapped
from action_space_toolbox.scripts.render import Renderer
from action_space_toolbox.tuning.controller_evaluator import ControllerEvaluator
from action_space_toolbox.util.angles import normalize_angle


class PositionControlEvaluator(ControllerEvaluator):
    def __init__(
        self,
        env_id: str,
        num_targets: int,
        repetitions_per_target: int,
        max_steps_per_episode: Optional[int],
    ):
        super().__init__(
            env_id,
            num_targets,
            repetitions_per_target,
        )
        self.max_steps_per_episode = max_steps_per_episode

    def _sample_targets(self, num_targets: int):
        # Unwrap the env to get back the original action space
        env = gym.make(self.env_id, normalize=False)
        self._prepare_env(env)
        env.seed(42)
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

    def visualize_targets(self) -> None:
        env = gym.make(self.env_id)
        self._prepare_env(env)
        renderer = Renderer()
        for pos in self.targets:
            env.reset()
            if isinstance(env.unwrapped, gym.envs.mujoco.MujocoEnv):
                full_pos = env.sim.data.qpos
                full_pos[env.actuated_joints] = pos
                env.set_state(full_pos, np.zeros_like(full_pos))
            elif isinstance(env.unwrapped, dmc2gym.wrappers.DMCWrapper):
                full_pos = env.physics.data.qpos
                full_pos[env.actuated_joints] = pos
                env.physics.set_state(
                    np.concatenate((full_pos, np.zeros_like(full_pos)))
                )
                env.step(np.zeros(env.action_space.shape))
            else:
                env.unwrapped.state = (pos, np.zeros_like(pos))
            renderer.render_frame(env)
            time.sleep(1)

    def evaluate_gains(
        self,
        gains: np.ndarray,
        render: bool = False,
    ) -> float:
        env = gym.make(
            self.env_id,
            p_gains=gains["p_gains"],
            d_gains=gains["d_gains"],
            normalize=False,
        )
        self._prepare_env(env)
        assert check_wrapped(env, FixedGainsPositionControlWrapper)
        joint_errors = []
        renderer = Renderer()
        for target_actuator_position in self.targets:
            for repetition in range(self.repetitions_per_target):
                done = False
                env.reset()
                joint_errors.append(0.0)
                i = 0
                while not done:
                    _, _, done, _ = env.step(target_actuator_position)
                    if render:
                        renderer.render_frame(env)
                    diff = env.actuator_positions - target_actuator_position
                    joint_diff = diff * ~np.array(
                        env.actuators_revolute
                    ) + normalize_angle(diff) * np.array(env.actuators_revolute)
                    joint_errors[-1] += np.mean(np.abs(joint_diff))
                    i += 1
                    if (
                        self.max_steps_per_episode is not None
                        and i >= self.max_steps_per_episode
                    ):
                        break
                joint_errors[-1] /= i

        return np.mean(joint_errors).item()
