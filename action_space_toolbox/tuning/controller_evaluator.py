from typing import Any, Dict

import numpy as np


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
