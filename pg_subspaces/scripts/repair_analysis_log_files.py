import argparse
import logging
import re
import subprocess
from pathlib import Path
from typing import Optional

import hydra
import yaml
from tqdm import tqdm

from pg_subspaces.metrics.tensorboard_logs import (
    create_event_accumulators,
    read_scalar,
)

logger = logging.getLogger(__name__)


def repair(
    run_dir: Path,
    expected_interval: int,
    last_checkpoint: Optional[int],
    analysis_run_id: str,
) -> None:
    hc_key = f"high_curvature_subspace_analysis/{analysis_run_id}/gradient_subspace_fraction_100evs/true_gradient/value_function_loss"
    analyses_yaml_path = run_dir / ".analyses.yaml"
    with analyses_yaml_path.open("r") as analyses_yaml_file:
        analyses_dict = yaml.safe_load(analyses_yaml_file)
    hc_analysis_steps = analyses_dict.get("high_curvature_subspace_analysis", {}).get(
        analysis_run_id, []
    )
    if last_checkpoint is None:
        if len(hc_analysis_steps) > 0:
            last_checkpoint = max(hc_analysis_steps)
        else:
            last_checkpoint = 3000000
    missing_steps = [
        step
        for step in range(0, last_checkpoint + 1, expected_interval)
        if step not in hc_analysis_steps
    ]
    steps_to_add = []
    if len(missing_steps) == 0:
        return
    else:
        ea = create_event_accumulators([run_dir / "tensorboard"])[0][1]
        if hc_key in ea.Tags()["scalars"]:
            scalars = read_scalar(ea, hc_key)
            for step in missing_steps:
                if step in scalars:
                    steps_to_add.append(step)
        with analyses_yaml_path.open("r") as analyses_yaml_file:
            analyses_dict = yaml.safe_load(analyses_yaml_file)
        analyses_dict["high_curvature_subspace_analysis"] = analyses_dict.get(
            "high_curvature_subspace_analysis", {}
        )
        hc_dict = analyses_dict["high_curvature_subspace_analysis"]
        hc_dict[analysis_run_id] = hc_dict.get(analysis_run_id, {})
        a_id_list = hc_dict[analysis_run_id]
        a_id_list.extend(steps_to_add)
        a_id_list.sort()
        with analyses_yaml_path.open("w") as analyses_yaml_file:
            yaml.dump(analyses_dict, analyses_yaml_file)
        print(f"Missing steps: {missing_steps}")
        print(f"Added steps: {steps_to_add}")


def get_step_from_checkpoint(file_name: str) -> int:
    step_str = max(re.findall("[0-9]*", file_name))
    return int(step_str)


def repair_analysis_log_files(
    train_logs: Path,
    log_dir: Path,
    expected_interval: int,
    last_checkpoint: Optional[int],
    analysis_run_id: str,
    sync_train_logs: bool,
) -> None:
    result_commit = subprocess.run(
        ["git", "-C", f"{Path(__file__).parent}", "rev-parse", "HEAD"],
        stdout=subprocess.PIPE,
    )
    logger.debug(
        f"Commit {Path(__file__).parents[2].name}: {result_commit.stdout.decode().strip()}"
    )
    train_logs = Path(train_logs)
    if not train_logs.is_absolute():
        train_logs = Path(hydra.utils.get_original_cwd()) / train_logs
    logger.info(f"Analyzing results in {train_logs}")

    experiment_dir = train_logs.parent if train_logs.name.isnumeric() else train_logs
    # assert re.match("[0-9]{2}-[0-9]{2}-[0-9]{2}", experiment_dir.name)
    train_logs_relative = train_logs.relative_to(experiment_dir.parents[3])
    train_logs_local = log_dir / train_logs_relative
    if sync_train_logs:
        train_logs_local.mkdir(exist_ok=True, parents=True)
        logger.info(f"Syncing logs from {train_logs} to {train_logs_local.absolute()}.")
        rsync_result = subprocess.run(
            ["rsync", "-uaz", str(train_logs) + "/", str(train_logs_local)],
            stderr=subprocess.STDOUT,
        )
        rsync_result.check_returncode()
        logger.info("Synced logs successfully.")
    else:
        train_logs_local = train_logs

    if (train_logs_local / "checkpoints").exists():
        run_dirs = [train_logs_local]
    else:
        run_dirs = [
            d for d in train_logs_local.iterdir() if d.is_dir() and d.name.isdigit()
        ]

    for run_dir in tqdm(run_dirs, total=len(run_dirs)):
        repair(run_dir, expected_interval, last_checkpoint, analysis_run_id)

    if sync_train_logs:
        logger.info(
            f"Syncing results from {train_logs_local} back to {train_logs.absolute()}."
        )
        rsync_result = subprocess.run(
            ["rsync", "-uaz", str(train_logs_local) + "/", str(train_logs)],
            stderr=subprocess.STDOUT,
        )
        rsync_result.check_returncode()
        logger.info("Synced results successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("train_logs", type=str)
    parser.add_argument("log_dir", type=str)
    parser.add_argument("--expected-interval", type=int, default=50000)
    parser.add_argument("--last-checkpoint", type=int)
    parser.add_argument("--analysis-run-id", type=str, default="default")
    parser.add_argument("--sync-logs", action="store_true")
    args = parser.parse_args()

    repair_analysis_log_files(
        Path(args.train_logs),
        Path(args.log_dir),
        args.expected_interval,
        args.last_checkpoint,
        args.analysis_run_id,
        args.sync_logs,
    )
