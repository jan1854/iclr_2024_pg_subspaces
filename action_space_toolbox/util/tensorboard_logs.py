import logging
import time
from pathlib import Path
from typing import Sequence, Tuple, Optional, Union, Any, Dict

import PIL.Image
import numpy as np
from tensorboard.backend.event_processing import event_accumulator
from torch.utils.tensorboard import SummaryWriter


class TensorboardLogs:
    """
    Tensorboard SummaryWriters are not serializable which makes it impossible to pass them around to different
    processes. Creating a new SummaryWriter in each process can result in a large number of event files. This class
    is serializable and thus provides a way to pass tensorboard logs between processes (to log them with a single
    SummaryWriter later on).
    """

    def __init__(self):
        self.scalars = {}
        self.images = {}
        self.custom_scalars_layout = {}

    def add_scalar(
        self, key: str, value: float, step: int, walltime: Optional[float] = None
    ) -> None:
        if walltime is None:
            walltime = time.time()
        if key in self.scalars:
            self.scalars[key].append((value, step, walltime))
        else:
            self.scalars[key] = [(value, step, walltime)]

    def add_image(
        self,
        key: str,
        image: Union[np.ndarray, PIL.Image.Image],
        step: int,
        walltime: Optional[float] = None,
    ) -> None:
        if walltime is None:
            walltime = time.time()
        if isinstance(image, PIL.Image.Image):
            image = image.convert("RGB")
            image = np.array(image)
        if key in self.images:
            self.images[key].append((image, step, walltime))
        else:
            self.images[key] = [(image, step, walltime)]

    def add_custom_scalars(self, layout: Dict[str, Any]) -> None:
        self.custom_scalars_layout.update(layout)

    def log(self, summary_writer: SummaryWriter) -> None:
        for key, scalars in self.scalars.items():
            for value, step, walltime in scalars:
                summary_writer.add_scalar(key, value, step, walltime)
        for key, images in self.images.items():
            for image, step, walltime in images:
                summary_writer.add_image(key, image, step, walltime, dataformats="HWC")
        summary_writer.add_custom_scalars(self.custom_scalars_layout)

    def update(self, logs: "TensorboardLogs") -> None:
        self.scalars.update(logs.scalars)
        self.images.update(logs.images)
        self.custom_scalars_layout.update(logs.custom_scalars_layout)


def get_output_keys(key: str, key_prefix: Optional[str]) -> Tuple[str, str, str]:
    if key_prefix is not None:
        key = f"{key_prefix}/{key}"
    return key, f"zz_confidence/{key}/mean-std", f"zz_confidence/{key}/mean+std"


def add_custom_scalar_layout(
    summary_writer: SummaryWriter, keys: Sequence[str], key_prefix: Optional[str]
) -> None:
    layout = {}
    for key in keys:
        key_split = key.split("/")
        section = key_split[0]
        title = "/".join(key_split[1:])
        if section not in layout:
            layout[section] = {}
        layout[section][title] = ["Margin", get_output_keys(key, key_prefix)]
    summary_writer.add_custom_scalars(layout)


def calculate_mean_std_sequence(
    event_accumulators: Sequence[event_accumulator.EventAccumulator], key: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Sort event_accumulators and check whether the steps match
    event_accumulators = sorted(event_accumulators, key=lambda e: len(e.Scalars(key)))
    steps = [scalar.step for scalar in event_accumulators[0].Scalars(key)]
    if len(steps) < len(event_accumulators[-1].Scalars(key)):
        logging.warning(
            f"Found a different number of scalars for the event accumulators (key: {key}, "
            f"min: {len(steps)}, max: {len(event_accumulators[-1].Scalars(key))}), using the minimum value"
        )

    steps = np.array([scalar.step for scalar in event_accumulators[0].Scalars(key)])
    values = np.array(
        [
            [ea.Scalars(key)[i].value for i in range(len(steps))]
            for ea in event_accumulators
        ]
    )
    wall_time = np.array(
        [
            [ea.Scalars(key)[i].wall_time for i in range(len(steps))]
            for ea in event_accumulators
        ]
    )

    return (
        steps,
        np.mean(wall_time, axis=0),
        np.mean(values, axis=0),
        np.std(values, axis=0),
    )


def add_mean_std_scalars(
    event_accumulators: Sequence[event_accumulator.EventAccumulator],
    summary_writer: SummaryWriter,
    key: str,
    output_key_prefix: Optional[str],
) -> None:
    # Sort event_accumulators and check whether the steps match
    event_accumulators = sorted(event_accumulators, key=lambda e: len(e.Scalars(key)))
    steps = [scalar.step for scalar in event_accumulators[0].Scalars(key)]
    if len(steps) < len(event_accumulators[-1].Scalars(key)):
        logging.warning(
            f"Found a different number of scalars for the event accumulators (key: {key}, "
            f"min: {len(steps)}, max: {len(event_accumulators[-1].Scalars(key))}), using the minimum value"
        )

    key_mean, key_lower_margin, key_upper_margin = get_output_keys(
        key, output_key_prefix
    )

    for step, wall_time, value_mean, value_std in zip(
        *calculate_mean_std_sequence(event_accumulators, key)
    ):
        summary_writer.add_scalar(key_mean, value_mean, step, wall_time)
        summary_writer.add_scalar(
            key_lower_margin, value_mean - value_std, step, wall_time
        )
        summary_writer.add_scalar(
            key_upper_margin, value_mean + value_std, step, wall_time
        )


def create_event_accumulators(tb_dirs: Sequence[Path]):
    event_accumulators = []
    for tb_dir in tb_dirs:
        ea = event_accumulator.EventAccumulator(str(tb_dir))
        ea.Reload()
        assert (
            len(ea.Tags()["scalars"]) > 0
        ), f"Log files in directory {tb_dir} contain no data."
        event_accumulators.append(ea)
    return event_accumulators


def combine_tb_logs(
    event_file_dirs: Sequence[Path],
    out_path: Path,
    output_key_prefix: Optional[str] = None,
    event_file_suffix: Optional[str] = "",
    existing_dir_ok: bool = False,
) -> None:
    assert (
        existing_dir_ok or not out_path.exists()
    ), f"Output directory {out_path} already exists"
    assert (
        out_path.parent.exists()
    ), f"Parent directory {out_path.parent} does not exist"

    event_accumulators = create_event_accumulators(event_file_dirs)

    out_path.mkdir(exist_ok=existing_dir_ok)
    summary_writer = SummaryWriter(str(out_path), filename_suffix=event_file_suffix)
    add_custom_scalar_layout(
        summary_writer, event_accumulators[0].Tags()["scalars"], output_key_prefix
    )
    tags = set()
    for ea in event_accumulators:
        tags.update(ea.Tags()["scalars"])
    for key in sorted(list(tags)):
        key_is_in_all_ea = True
        for ea in event_accumulators:
            if key not in ea.Tags()["scalars"]:
                key_is_in_all_ea = False
                break
        if not key_is_in_all_ea:
            logging.warning(
                f"Did not find key {key} in all event accumulators. Skipping."
            )
            continue
        if key.startswith("zz_confidence"):
            continue
        add_mean_std_scalars(event_accumulators, summary_writer, key, output_key_prefix)
