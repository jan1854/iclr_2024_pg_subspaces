import json
from pathlib import Path
from typing import Dict

import numpy as np
from tqdm import trange

from action_space_toolbox.tuning.controller_evaluator import ControllerEvaluator


def _gains_dict_to_str(gains_dict: Dict[str, np.ndarray]):
    return ", ".join(
        [f"{name}: {gains.tolist()}" for name, gains in gains_dict.items()]
    )


def tune_controller_gains(
    evaluator: ControllerEvaluator,
    num_iterations: int,
    gains_exp_limits: Dict[str, np.ndarray],
    log_path: Path,
    render: bool = False,
) -> Dict[str, np.ndarray]:
    best_loss = float("inf")
    best_gains = None
    progress_bar = trange(num_iterations)
    losses = []
    for i in progress_bar:
        gains = {
            name: np.around(
                10 ** np.random.uniform(exp_limits[:, 0], exp_limits[:, 1]), decimals=2
            )
            for name, exp_limits in gains_exp_limits.items()
        }
        loss = evaluator.evaluate_gains(gains, render=render)
        losses.append(loss)
        if loss < best_loss:
            best_gains = gains
            best_loss = loss
        progress_bar.set_description(
            f"Tuning gains. Best gains: {_gains_dict_to_str(best_gains)}, "
            f"loss: {best_loss:.4f}, average_loss: {np.mean(losses):.4f}."
        )

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log_file:
        json.dump(
            {
                "gains": {name: gains.tolist() for name, gains in best_gains.items()},
                "loss": np.around(best_loss, decimals=2),
            },
            log_file,
        )

    return best_gains
