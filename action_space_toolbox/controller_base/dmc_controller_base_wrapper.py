import dmc2gym.wrappers
import numpy as np

from action_space_toolbox.controller_base.controller_base_wrapper import (
    ControllerBaseWrapper,
)


class DMCControllerBaseWrapper(ControllerBaseWrapper):
    def __init__(self, env: dmc2gym.wrappers.DMCWrapper):
        assert isinstance(env.unwrapped, dmc2gym.wrappers.DMCWrapper)
        self.actuated_joints = env.physics.model.actuator_trnid[:, 0]
        actuators_revolute = env.physics.model.jnt_type == 3
        actuator_pos_bounds = []
        for j in self.actuated_joints:
            if env.physics.model.jnt_limited[j]:
                actuator_pos_bounds.append(env.physics.model.jnt_range[j])
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
        return self.env.physics.data.qpos[self.actuated_joints]

    @property
    def actuator_velocities(self) -> np.ndarray:
        return self.env.physics.data.qvel[self.actuated_joints]

    @property
    def timestep(self) -> float:
        return self.env.physics.model.opt.timestep

    def set_timestep(self, timestep: float) -> None:
        self.env.physics.model.opt.timestep = timestep
