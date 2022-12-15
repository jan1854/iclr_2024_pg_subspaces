import random
import time
from typing import List, Tuple, Optional

import dmc2gym.wrappers
import gym
import gym.envs.mujoco
import numpy as np
from dm_control import mjcf

from action_space_toolbox.control_modes.position_control.fixed_gains_position_control_wrapper import (
    FixedGainsPositionControlWrapper,
)
from action_space_toolbox.control_modes.check_wrapped import check_wrapped
from action_space_toolbox.scripts.render import Renderer
from action_space_toolbox.tuning.controller_evaluator import ControllerEvaluator
from action_space_toolbox.util.angles import normalize_angle

State = Tuple[np.ndarray, np.ndarray]


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

    def _sample_targets(self, num_targets: int) -> List[Tuple[State, np.ndarray]]:
        # Unwrap the env to get back the original action space
        env = gym.make(self.env_id, normalize=False)
        env.seed(42)
        self._prepare_env(env)
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

        return np.mean(joint_errors)  # type: ignore

    @staticmethod
    def _set_state(env, state: State) -> None:
        if isinstance(env.unwrapped, gym.envs.mujoco.MujocoEnv):
            env.set_state(*state)
        elif isinstance(env.unwrapped, dmc2gym.wrappers.DMCWrapper):
            env.physics.set_state(state)
        else:
            env.unwrapped.state = state

    @staticmethod
    def _prepare_env(env: gym.Env) -> None:
        if (
            env.spec.id == "Pendulum_PC-v1"
            or env.spec.id == "Reacher_PC-v2"
            or env.spec.id.startswith("dmc_Finger")
            or env.spec.id.startswith("dmc_Reacher")
        ):
            pass
        elif (
            env.spec.id.startswith("dmc_Cheetah")
            or env.spec.id.startswith("dmc_Hopper")
            or env.spec.id.startswith("dmc_Walker")
        ):
            if env.spec.id.startswith("dmc_Cheetah"):
                import dm_control.suite.cheetah as task_module
            elif env.spec.id.startswith("dmc_Hopper"):
                import dm_control.suite.hopper as task_module
            elif env.spec.id.startswith("dmc_Walker"):
                import dm_control.suite.walker as task_module
            else:
                raise ValueError()
            model, assets = task_module.get_model_and_assets()
            model = mjcf.from_xml_string(model, assets=assets)
            model.find("body", "torso").pos = np.array([0.0, 0.0, 2.0])
            # Increase stiffness and damping for the root joints to make them quasi static
            for joint_name in ["rootx", "rootz", "rooty"]:
                joint = model.find("joint", joint_name)
                joint.stiffness = 10000.0
                joint.damping = 100000.0

            physics = task_module.Physics.from_xml_string(model.to_xml_string())
            env.unwrapped._env._physics = physics
        else:
            raise ValueError(f"Unsupported environment: {env.unwrapped.spec.id}")
