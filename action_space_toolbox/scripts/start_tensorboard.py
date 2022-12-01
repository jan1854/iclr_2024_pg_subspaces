#!/usr/bin/env python3
import tempfile
from argparse import ArgumentParser
from pathlib import Path
from threading import Event

import tensorboard


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("logdirs", type=str, nargs="+")
    args = parser.parse_args()

    tb_dirs_to_display = {}
    for logdir in args.logdirs:
        if ":" in logdir:
            new_path_rel, logdir = logdir.split(":")
            new_path_rel = Path(new_path_rel)
        else:
            new_path_rel = None
        logdir = Path(logdir)
        if (logdir / "tensorboard").exists():
            tb_dirs = [logdir / "tensorboard"]
        elif (logdir / "combined").exists():
            tb_dirs = [logdir / "combined"]
        else:
            tb_dirs = list(logdir.glob("*/tensorboard"))
        if len(tb_dirs) == 0:
            tb_dirs = [logdir]
        for tb_dir in tb_dirs:
            if (tb_dir / "trainings").exists():
                best_training_tb_dir = tb_dir / "trainings" / "best_config" / "combined"
            else:
                best_training_tb_dir = tb_dir
            assert (
                len(list(best_training_tb_dir.glob("events.out.tfevents*"))) > 0
            ), f"Did not find any tensorboard log files in {best_training_tb_dir}"
            time_dir = tb_dir.parent if tb_dir.name == "tensorboard" or tb_dir.name == "combined" else tb_dir
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
