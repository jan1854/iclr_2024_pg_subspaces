import random
import time
from typing import List, Tuple

import dmc2gym.wrappers
import gym
import gym.envs.mujoco
import numpy as np

import action_space_toolbox
from action_space_toolbox.control_modes.velocity_control.fixed_gains_velocity_control_wrapper import (
    FixedGainsVelocityControlWrapper,
)
from action_space_toolbox.control_modes.check_wrapped import check_wrapped
from action_space_toolbox.tuning.controller_evaluator import ControllerEvaluator


State = Tuple[np.ndarray, np.ndarray]


class VelocityControlEvaluator(ControllerEvaluator):
    def __init__(
        self,
        env_id: str,
        num_targets: int,
        repetitions_per_target: int,
        num_episodes_to_sample_targets_from: int = 50,
    ):
        self.num_episodes_to_sample_targets_from = num_episodes_to_sample_targets_from
        super().__init__(
            env_id,
            num_targets,
            repetitions_per_target,
        )

    def _sample_targets(self, num_targets: int) -> List[Tuple[State, np.ndarray]]:
        # Unwrap the env to get back the original action space
        env = gym.make(self.env_id, normalize=False)
        self._prepare_env(env)
        env.seed(42)
        tmp_env = env
        while not isinstance(tmp_env.env, FixedGainsVelocityControlWrapper):
            tmp_env = tmp_env.env
        tmp_env.env = tmp_env.env.env
        acturator_velocities = []
        for _ in range(self.num_episodes_to_sample_targets_from):
            env.reset()
            done = False
            actuator_velocities_curr_episode = []
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
                actuator_velocities_curr_episode.append(
                    (state, env.actuator_velocities)
                )
            acturator_velocities.extend(actuator_velocities_curr_episode)
        return random.choices(acturator_velocities, k=num_targets)

    def visualize_targets(self) -> None:
        env = gym.make(self.env_id)
        self._prepare_env(env)
        env.reset()
        for target in self.targets:
            self._set_state(env, target[0])
            env.render()
        time.sleep(2)

    def evaluate_gains(
        self,
        gains: np.ndarray,
        render: bool = False,
    ) -> float:
        env = gym.make(self.env_id, gains=gains["gains"], normalize=False)
        self._prepare_env(env)
        env.seed(23)
        assert check_wrapped(env, FixedGainsVelocityControlWrapper)
        joint_vel_errors = []
        for target in self.targets:
            for _ in range(self.repetitions_per_target):
                env.reset()
                self._set_state(env, target[0])
                env.step(target[1])
                if render:
                    env.render()
                diff = env.actuator_velocities - target[1]
                joint_vel_errors.append(np.mean(np.abs(diff)))

        return np.mean(joint_vel_errors)  # type: ignore

    @staticmethod
    def _set_state(env, state: State) -> None:
        if isinstance(env.unwrapped, gym.envs.mujoco.MujocoEnv):
            env.set_state(*state)
        elif isinstance(env.unwrapped, dmc2gym.wrappers.DMCWrapper):
            env.physics.set_state(state)
        else:
            env.unwrapped.state = state
