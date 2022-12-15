from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

import gym.envs.mujoco
import numpy as np

from action_space_toolbox.tuning.position_control_evaluator import (
    PositionControlEvaluator,
)
from action_space_toolbox.tuning.tune_controller_gains import tune_controller_gains


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("env_id", type=str)
    parser.add_argument("num_iterations", type=int)
    parser.add_argument("p_gains_exp_low", type=float)
    parser.add_argument("p_gains_exp_high", type=float)
    parser.add_argument("d_gains_exp_low", type=float)
    parser.add_argument("d_gains_exp_high", type=float)
    parser.add_argument("--visualize-targets", action="store_true")
    parser.add_argument("--targets-to-sample", type=int, default=25)
    parser.add_argument("--repetitions-per-target", type=int, default=1)
    parser.add_argument("--max-steps-per-episode", type=int, default=200)
    args = parser.parse_args()

    evaluator = PositionControlEvaluator(
        args.env_id,
        args.targets_to_sample,
        args.repetitions_per_target,
        max_steps_per_episode=args.max_steps_per_episode,
    )

    if args.visualize_targets:
        evaluator.visualize_targets()

    tmp_env = gym.make(args.env_id)
    assert len(tmp_env.action_space.shape) == 1
    action_size = tmp_env.action_space.shape[0]
    gain_limits = {
        "p_gains": np.array(
            [[args.p_gains_exp_low, args.p_gains_exp_high] for _ in range(action_size)]
        ),
        "d_gains": np.array(
            [[args.d_gains_exp_low, args.d_gains_exp_high] for _ in range(action_size)]
        ),
    }
    log_path = (
        Path(__file__).parents[2]
        / "logs"
        / "tuning_results"
        / "pc"
        / f"{args.env_id}_{datetime.now().strftime('%y-%m-%d_%H-%M-%S')}.json"
    )
    tuned_gains = tune_controller_gains(
        evaluator, args.num_iterations, gain_limits, log_path
    )
