#!/usr/bin/env python3
import os
import tempfile
from argparse import ArgumentParser
from pathlib import Path
from threading import Event

import tensorboard
import yaml

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("rundirs", type=str, nargs="*", default=[])
    parser.add_argument("--logdir", type=str)
    parser.add_argument("--config", type=str, action="append", default=[])
    args = parser.parse_args()

    if args.logdir is not None:
        training_dir = Path(args.logdir) / "training"
    else:
        if "ACTION_SPACE_TOOLBOX_LOG_DIR" in os.environ:
            log_dir = Path(os.environ["ACTION_SPACE_TOOLBOX_LOG_DIR"])
        else:
            log_dir = Path(__file__).parents[2] / "logs"
        training_dir = log_dir / "training"

    rundirs = []
    for config_name in args.config:
        if not config_name.endswith(".yaml"):
            config_name = config_name + ".yaml"
        config_path = Path(__file__).parent / "res" / "tb_configs" / config_name
        with config_path.open("r") as config_file:
            rundirs.extend(
                [
                    (new_path_rel, training_dir / rundir)
                    for new_path_rel, rundir in yaml.safe_load(config_file).items()
                ]
            )

    for rundir in args.rundirs:
        if ":" in rundir:
            new_path_rel, rundir = rundir.split(":")
        else:
            new_path_rel = None
        rundirs.append((new_path_rel, Path(rundir)))

    tb_dirs_to_display = {}
    for new_path_rel, rundir in rundirs:
        if (rundir / "tensorboard").exists():
            tb_dirs = [rundir / "tensorboard"]
        elif (rundir / "combined").exists():
            tb_dirs = [rundir / "combined"]
        else:
            tb_dirs = list(rundir.glob("*/tensorboard"))
        if len(tb_dirs) == 0:
            tb_dirs = [rundir]
        for tb_dir in tb_dirs:
            if (tb_dir / "trainings").exists():
                best_training_tb_dir = tb_dir / "trainings" / "best_config" / "combined"
            else:
                best_training_tb_dir = tb_dir
            assert (
                len(list(best_training_tb_dir.glob("events.out.tfevents*"))) > 0
            ), f"Did not find any tensorboard log files in {best_training_tb_dir}"
            time_dir = (
                tb_dir.parent
                if tb_dir.name == "tensorboard" or tb_dir.name == "combined"
                else tb_dir
            )
            if new_path_rel is None:
                new_path_rel = time_dir.relative_to(time_dir.parent.parent.parent)
            tb_dirs_to_display[new_path_rel] = best_training_tb_dir

    # Build a symlink tree with the log directories since tensorboard does not support selecting only some logs (see
    # https://github.com/tensorflow/tensorboard/issues/5438#issuecomment-982048444)
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        print(f"Symlink tree located in {temp_dir}")
        for new_path_rel, dest in tb_dirs_to_display.items():
            symlink_path = temp_dir / new_path_rel
            symlink_path.parent.mkdir(parents=True, exist_ok=True)
            symlink_path.symlink_to(dest, target_is_directory=True)

        tb = tensorboard.program.TensorBoard()
        tb.configure(argv=[None, "--logdir", str(temp_dir)])
        url = tb.launch()
        print(f"Tensorflow listening on {url}")
        Event().wait()
