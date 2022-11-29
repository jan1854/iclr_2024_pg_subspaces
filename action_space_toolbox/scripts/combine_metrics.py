import argparse
import shutil
from pathlib import Path
from typing import Sequence, Any, Dict

import yaml
from tqdm import tqdm

from action_space_toolbox.util.tensorboard_logs import combine_tb_logs


def same_configurations(run_dirs: Sequence[Path]) -> bool:
    configs = []
    for run_dir in run_dirs:
        with (run_dir / ".hydra" / "config.yaml").open("r") as config_file:
            configs.append(yaml.safe_load(config_file))

    for config in configs[1:]:
        if not config_equal(configs[0], config, ["seed"]):
            return False
    return True


def config_equal(
    c1: Dict[str, Any], c2: Dict[str, Any], exception_keys: Sequence[str] = ()
) -> bool:
    if len(c1) != len(c2):
        return False
    for k, v1 in c1.items():
        if k not in c2:
            return False
        elif k not in exception_keys:
            if isinstance(v1, Dict):
                if not isinstance(c2[k], Dict) or not config_equal(
                    v1, c2[k], exception_keys
                ):
                    return False
            else:
                if v1 != c2[k]:
                    return False
    return True


def combine_metrics(log_path: Path) -> None:
    jobs = []
    training_log_path = log_path / "training"
    for env_dir in training_log_path.iterdir():
        for date_dir in env_dir.iterdir():
            for experiment_dir in date_dir.iterdir():
                run_dirs = [
                    d
                    for d in experiment_dir.iterdir()
                    if d.is_dir() and d.name.isnumeric()
                ]

                # Create the combined metrics if either the "combined" directory is missing or any of the runs contains
                # new data
                combined_dir = experiment_dir / "combined"
                update = not combined_dir.exists()
                for curr_run_dir in run_dirs:
                    if curr_run_dir.exists():
                        update |= (curr_run_dir / ".new_data").exists()
                    else:
                        break

                if len(run_dirs) > 1 and same_configurations(run_dirs) and update:
                    jobs.append((run_dirs, combined_dir))

    for run_dirs, combined_dir in tqdm(jobs):
        tb_dirs = [run_dir / "tensorboard" for run_dir in run_dirs]
        if combined_dir.exists():
            shutil.rmtree(combined_dir)
        combine_tb_logs(tb_dirs, combined_dir)
        for run_dir in run_dirs:
            (run_dir / ".new_data").unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("log_path", type=str)
    args = parser.parse_args()

    combine_metrics(Path(args.log_path))
