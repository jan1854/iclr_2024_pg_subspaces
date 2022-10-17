import gym
import numpy as np
from gym.envs.mujoco import MujocoEnv

from action_space_toolbox.dof_information.dof_information_wrapper import (
    DofInformationWrapper,
)


class MujocoDofInformationWrapper(DofInformationWrapper):
    def __init__(self, env: gym.Env):
        assert isinstance(env.unwrapped, MujocoEnv)
        self.actuated_joints = env.model.actuator_trnid[:, 0]
        dof_pos_bounds = (
            env.model.jnt_range[self.actuated_joints]
            if np.all(env.model.jnt_limited[self.actuated_joints])
            else None
        )
        dof_vel_bounds = None  # TODO: There does not appear to be a maximum joint velocity property of the Mujoco environments
        dofs_revolute = env.model.jnt_type[self.actuated_joints] == 3
        super().__init__(env, dof_pos_bounds, dof_vel_bounds, dofs_revolute)

    @property
    def dof_positions(self) -> np.ndarray:
        return np.array([self.env.sim.data.qpos[self.actuated_joints]])

    @property
    def dof_velocities(self) -> np.ndarray:
        return np.array([self.env.sim.data.qvel[self.actuated_joints]])
