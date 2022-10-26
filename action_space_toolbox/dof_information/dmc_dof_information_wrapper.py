import dmc2gym.wrappers
import numpy as np

from action_space_toolbox.dof_information.dof_information_wrapper import (
    DofInformationWrapper,
)


class DMCDofInformationWrapper(DofInformationWrapper):
    def __init__(self, env: dmc2gym.wrappers.DMCWrapper):
        assert isinstance(env.unwrapped, dmc2gym.wrappers.DMCWrapper)
        self.actuated_joints = env.physics.model.actuator_trnid[:, 0]
        dofs_revolute = env.physics.model.jnt_type == 3
        dof_pos_bounds = []
        for j in self.actuated_joints:
            if env.physics.model.jnt_limited[j]:
                dof_pos_bounds.append(env.physics.model.jnt_range[j])
            elif dofs_revolute[j]:
                dof_pos_bounds.append(np.array([-np.pi, np.pi]))
            else:
                dof_pos_bounds.append(np.array([-np.inf, np.inf]))
        dof_pos_bounds = np.array(dof_pos_bounds)
        dof_vel_bounds = None  # TODO: There does not appear to be a maximum joint velocity property of the Mujoco environments
        super().__init__(
            env, dof_pos_bounds, dof_vel_bounds, dofs_revolute[self.actuated_joints]
        )

    @property
    def dof_positions(self) -> np.ndarray:
        return self.env.physics.data.qpos[self.actuated_joints]

    @property
    def dof_velocities(self) -> np.ndarray:
        return self.env.physics.data.qvel[self.actuated_joints]
