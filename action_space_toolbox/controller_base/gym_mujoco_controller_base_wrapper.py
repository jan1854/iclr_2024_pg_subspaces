import gym
import numpy as np
from gym.envs.mujoco import MujocoEnv

from action_space_toolbox.controller_base.controller_base_wrapper import (
    ControllerBaseWrapper,
)


class GymMujocoControllerBaseWrapper(ControllerBaseWrapper):
    def __init__(self, env: gym.Env):
        assert isinstance(env.unwrapped, MujocoEnv)
        self.actuated_joints = env.model.actuator_trnid[:, 0]
        actuators_revolute = env.model.jnt_type == 3
        actuator_pos_bounds = []
        for j in self.actuated_joints:
            if env.model.jnt_limited[j]:
                actuator_pos_bounds.append(env.model.jnt_range[j])
            elif actuators_revolute[j]:
                actuator_pos_bounds.append(np.array([-np.pi, np.pi]))
            else:
                actuator_pos_bounds.append(np.array([-np.inf, np.inf]))
        actuator_pos_bounds = np.array(actuator_pos_bounds)
        actuator_vel_bounds = None  # TODO: There does not appear to be a maximum joint velocity property of the Mujoco environments
        super().__init__(
            env,
            actuator_pos_bounds,
            actuator_vel_bounds,
            actuators_revolute[self.actuated_joints],
        )

    @property
    def actuator_positions(self) -> np.ndarray:
        return self.env.sim.data.qpos[self.actuated_joints]

    @property
    def actuator_velocities(self) -> np.ndarray:
        return self.env.sim.data.qvel[self.actuated_joints]

    @property
    def timestep(self) -> float:
        return self.env.sim.model.opt.timestep

    def set_timestep(self, timestep: float) -> None:
        self.env.sim.model.opt.timestep = timestep
