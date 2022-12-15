from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

import gym.envs.mujoco
import numpy as np

from action_space_toolbox.tuning.tune_controller_gains import tune_controller_gains
from action_space_toolbox.tuning.velocity_control_evaluator import (
    VelocityControlEvaluator,
)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("env_id", type=str)
    parser.add_argument("num_iterations", type=int)
    parser.add_argument("gains_exp_low", type=float)
    parser.add_argument("gains_exp_high", type=float)
    parser.add_argument("--visualize-targets", action="store_true")
    parser.add_argument("--targets-to-sample", type=int, default=1000)
    parser.add_argument("--repetitions-per-target", type=int, default=1)
    args = parser.parse_args()

    evaluator = VelocityControlEvaluator(
        args.env_id, args.targets_to_sample, args.repetitions_per_target
    )

    if args.visualize_targets:
        evaluator.visualize_targets()

    tmp_env = gym.make(args.env_id)
    assert len(tmp_env.action_space.shape) == 1
    action_size = tmp_env.action_space.shape[0]
    gain_limits = {
        "gains": np.array(
            [[args.gains_exp_low, args.gains_exp_high] for _ in range(action_size)]
        )
    }
    log_path = (
        Path(__file__).parents[2]
        / "logs"
        / "tuning_results"
        / "vc"
        / f"{args.env_id}_{datetime.now().strftime('%y-%m-%d_%H-%M-%S')}.json"
    )
    tuned_gains = tune_controller_gains(
        evaluator, args.num_iterations, gain_limits, log_path
    )
