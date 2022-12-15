from typing import Dict

import gym
import numpy as np
from dm_control import mjcf


# TODO: This should use generics for the State
class ControllerEvaluator:
    def __init__(
        self,
        env_id: str,
        num_targets: int,
        repetitions_per_target: int,
    ):
        self.env_id = env_id
        self.targets = self._sample_targets(num_targets)
        self.repetitions_per_target = repetitions_per_target

    def _sample_targets(self, num_targets: int):
        pass

    def visualize_targets(self) -> None:
        pass

    def evaluate_gains(
        self,
        gains: Dict[str, np.ndarray],
        render: bool = False,
    ) -> float:
        pass

    @staticmethod
    def _prepare_env(env: gym.Env) -> None:
        if (
            env.spec.id.startswith("Pendulum")
            or env.spec.id.startswith("Reacher")
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
