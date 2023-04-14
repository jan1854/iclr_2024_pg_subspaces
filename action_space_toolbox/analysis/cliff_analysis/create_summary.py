import argparse
import csv
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np
import yaml

from action_space_toolbox.util.metrics import mean_relative_difference


def cliff_criterion(
    reward_checkpoint_undiscounted: float,
    reward_cliff_test_undiscounted: float,
    global_reward_range: float,
    cliff_reward_decrease: float = 0.5,
    cliff_reward_decrease_global: float = 0.25,
) -> bool:
    reward_checkpoint = np.mean(reward_checkpoint_undiscounted)
    reward_cliff_test = np.mean(reward_cliff_test_undiscounted)
    return (
        reward_cliff_test <= cliff_reward_decrease * reward_checkpoint
        and (reward_checkpoint - reward_cliff_test) / global_reward_range
        > cliff_reward_decrease_global
    )


def dump_results(experiment_dir: Path, results: Dict) -> None:
    out_dir = experiment_dir / "combined" / "cliff_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    for analysis_run_id, results_id in results.items():
        with (out_dir / f"results_{analysis_run_id}.csv").open("w") as csvfile:
            with (out_dir / f"additional_information_{analysis_run_id}.txt").open(
                "w"
            ) as infofile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow(
                    [
                        "Algorithm",
                        "Configuration",
                        "Rel. Reward change (cliff)",
                        "Rel. Reward change (no cliff)",
                    ]
                )
                for algorithm_name, results_algo in results_id.items():
                    # Sort the output but always put default as first item
                    results_algo_items_sorted = [
                        ("default", results_algo["default"])
                    ] + sorted([r for r in results_algo.items() if r[0] != "default"])
                    for config_str, results_config in results_algo_items_sorted:
                        reward_change_cliff = (
                            f"{np.mean(results_config['cliff']):.6f}"
                            if len(results_config["cliff"]) > 0
                            else "N/A"
                        )
                        reward_change_no_cliff = (
                            f"{np.mean(results_config['no cliff']):.6f}"
                            if len(results_config["no cliff"]) > 0
                            else "N/A"
                        )
                        csvwriter.writerow(
                            [
                                algorithm_name,
                                config_str,
                                reward_change_cliff,
                                reward_change_no_cliff,
                            ]
                        )
                infofile.write(
                    f"cliff locations: {len(results_config['cliff'])}\n"
                    f"non-cliff locations: {len(results_config['no cliff'])}"
                )


def append_sequence_dicts(dicts: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    keys = set()
    for d in dicts:
        keys.update(d.keys())
    results = {}
    for key in keys:
        value_type = None
        for d in dicts:
            if key in d:
                value_type = type(d[key])
                break
        if value_type is dict:
            results[key] = append_sequence_dicts([d[key] for d in dicts if key in d])
        else:
            new_result = []
            for d in dicts:
                if key in d:
                    new_result.extend(d[key])
            results[key] = new_result
    return results


def create_summary(experiment_dir: Path) -> None:
    run_dirs = [
        d for d in experiment_dir.iterdir() if d.is_dir() and d.name.isnumeric()
    ]
    if len(run_dirs) > 0:
        update_reward_changes = []
        for run_dir in run_dirs:
            update_reward_changes.append({})
            analyses_dir = run_dir / "analyses" / "cliff_analysis"
            for analysis_run_id in [d.name for d in analyses_dir.iterdir()]:
                if analysis_run_id not in update_reward_changes:
                    update_reward_changes[-1][analysis_run_id] = {}
                update_reward_changes_id = update_reward_changes[-1][analysis_run_id]
                results_path = analyses_dir / analysis_run_id / "results.yaml"
                with results_path.open("r") as f:
                    results = yaml.safe_load(f)
                min_reward = min(
                    [
                        res["reward_checkpoint"]["rewards_undiscounted"]
                        for res in results.values()
                    ]
                )
                max_reward = max(
                    [
                        res["reward_checkpoint"]["rewards_undiscounted"]
                        for res in results.values()
                    ]
                )
                global_reward_range = max_reward - min_reward
                for env_step, env_step_results in results.items():
                    reward_checkpoint = env_step_results["reward_checkpoint"][
                        "rewards_undiscounted"
                    ]
                    reward_cliff_test = env_step_results["reward_cliff_test"][
                        "rewards_undiscounted"
                    ]
                    for algorithm_name, algorithm_results in env_step_results[
                        "configs"
                    ].items():
                        if algorithm_name not in update_reward_changes_id:
                            update_reward_changes_id[algorithm_name] = {}
                        update_reward_changes_algo = update_reward_changes_id[
                            algorithm_name
                        ]
                        for config_name, config_results in algorithm_results.items():
                            if config_name not in update_reward_changes_algo:
                                update_reward_changes_algo[config_name] = {}
                            update_reward_changes_config = update_reward_changes_algo[
                                config_name
                            ]
                            is_cliff = cliff_criterion(
                                reward_checkpoint,
                                reward_cliff_test,
                                global_reward_range,
                            )
                            is_cliff_str = "cliff" if is_cliff else "no cliff"
                            if is_cliff_str not in update_reward_changes_config:
                                update_reward_changes_config[is_cliff_str] = []
                            curr_reward_change = mean_relative_difference(
                                reward_checkpoint,
                                np.array(
                                    config_results["reward_update"][
                                        "rewards_undiscounted"
                                    ]
                                ),
                            )
                            update_reward_changes_config[is_cliff_str].append(
                                curr_reward_change
                            )
        update_reward_changes = append_sequence_dicts(update_reward_changes)
        dump_results(experiment_dir, update_reward_changes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("log_path", type=str)

    args = parser.parse_args()

    create_summary(Path(args.log_path))