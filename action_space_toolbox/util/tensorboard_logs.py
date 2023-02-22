import io
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import PIL.Image
from tensorboard.backend.event_processing import event_accumulator
from torch.utils.tensorboard import SummaryWriter


logger = logging.getLogger(__name__)


class TensorboardLogs:
    """
    Tensorboard SummaryWriters are not serializable which makes it impossible to pass them around to different
    processes. Creating a new SummaryWriter in each process can result in a large number of event files. This class
    is serializable and thus provides a way to pass tensorboard logs between processes (to log them with a single
    SummaryWriter later on).
    """

    def __init__(
        self, prefix: Optional[str] = None, prefix_step_plots: Optional[str] = None
    ):
        self.scalars = {}
        self.images = {}
        self.custom_scalars_layout = {}
        self.prefix = prefix
        self.prefix_step_plots = prefix_step_plots

    def add_scalar(
        self, key: str, value: float, step: int, walltime: Optional[float] = None
    ) -> None:
        key = self._maybe_add_prefix(key, self.prefix)
        if walltime is None:
            walltime = time.time()
        self._add_scalar(key, value, step, walltime)

    def _add_scalar(self, key: str, value: float, step: int, walltime: float):
        if key in self.scalars:
            self.scalars[key].append((value, step, walltime))
        else:
            self.scalars[key] = [(value, step, walltime)]

    def add_step_plot(
        self,
        key: str,
        steps: Sequence[int],
        values: Sequence[float],
        walltimes: Optional[Sequence[float]] = None,
    ):
        key = self._maybe_add_prefix(key, self.prefix_step_plots)
        if walltimes is None:
            walltimes = [time.time()] * len(steps)
        assert len(steps) == len(values) == len(walltimes)
        for step, value, walltime in zip(steps, values, walltimes):
            self._add_scalar(key, value, step, walltime)

    def add_image(
        self,
        key: str,
        image: Union[np.ndarray, PIL.Image.Image],
        step: int,
        walltime: Optional[float] = None,
    ) -> None:
        key = self._maybe_add_prefix(key, self.prefix)
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
        if self.prefix is not None:
            layout = {self.prefix: layout}
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

    @classmethod
    def _maybe_add_prefix(cls, key: str, prefix: Optional[str]) -> str:
        if prefix is not None:
            return f"{prefix}/{key}"
        else:
            return key


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


def read_scalar(
    event_accumulator: event_accumulator.EventAccumulator, key: str
) -> Dict[int, event_accumulator.ScalarEvent]:
    warning_multiple_values_issued = False
    scalars = {}
    for scalar in event_accumulator.Scalars(key):
        if scalar.step in scalars:
            if not warning_multiple_values_issued:
                logger.warning(
                    f"Found multiple values for the same step for scalar {key}, using the most recent value."
                )
                warning_multiple_values_issued = True
            if scalar.wall_time > scalars[scalar.step].wall_time:
                scalars[scalar.step] = scalar
        else:
            scalars[scalar.step] = scalar
    return scalars


def calculate_mean_std_sequence(
    event_accumulators: Sequence[event_accumulator.EventAccumulator], key: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    assert len(event_accumulators) > 0
    scalars = []
    for ea in event_accumulators:
        scalars_curr_ea = read_scalar(ea, key)
        scalars.append(scalars_curr_ea)
    # Sort data and check whether the steps match
    scalars = sorted([list(s.values()) for s in scalars], key=lambda s: len(s))
    steps = set([s.step for s in scalars[-1]])
    for scalar in scalars:
        scalar.sort(key=lambda s: s.step)
        assert np.all(s.step in steps for s in scalar), "Steps do not match."
    steps = [scalar.step for scalar in scalars[0]]
    if len(steps) < len(scalars[-1]):
        logger.warning(
            f"Found a different number of scalars for the event accumulators (key: {key}, "
            f"min: {len(steps)}, max: {len(scalars[-1])}), using the minimum value"
        )

    values = np.array(
        [
            [scalars_curr_run[i].value for i in range(len(steps))]
            for scalars_curr_run in scalars
        ]
    )
    wall_time = np.array(
        [
            [scalars_curr_run[i].wall_time for i in range(len(steps))]
            for scalars_curr_run in scalars
        ]
    )

    return (
        np.array(steps),
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
        logger.warning(
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


def create_event_accumulators(
    tb_dirs: Sequence[Path],
) -> List[Tuple[Path, event_accumulator.EventAccumulator]]:
    event_accumulators = []
    for tb_dir in tb_dirs:
        ea = event_accumulator.EventAccumulator(str(tb_dir))
        ea.Reload()
        assert (
            len(ea.Tags()["scalars"]) > 0
        ), f"Log files in directory {tb_dir} contain no data."
        event_accumulators.append((tb_dir.parent, ea))
    return event_accumulators


# TODO: This should probably use the TensorboardLogs class
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
        summary_writer, event_accumulators[0][1].Tags()["scalars"], output_key_prefix
    )
    tags_scalars = set()
    for _, ea in event_accumulators:
        tags_scalars.update(ea.Tags()["scalars"])
    for key in sorted(list(tags_scalars)):
        key_is_in_all_ea = True
        for _, ea in event_accumulators:
            if key not in ea.Tags()["scalars"]:
                key_is_in_all_ea = False
                break
        if not key_is_in_all_ea:
            logger.warning(
                f"Did not find key {key} in all event accumulators. Skipping."
            )
            continue
        if key.startswith("zz_confidence"):
            continue
        add_mean_std_scalars(
            [ea for path, ea in event_accumulators],
            summary_writer,
            key,
            output_key_prefix,
        )
    for path, ea in event_accumulators:
        image_tags = ea.Tags()["images"]
        for tag in image_tags:
            for image_event in ea.Images(tag):
                with PIL.Image.open(
                    io.BytesIO(image_event.encoded_image_string)
                ) as image:
                    image_np = np.array(image)
                summary_writer.add_image(
                    f"{tag}/run_{path.name}",
                    image_np,
                    image_event.step,
                    image_event.wall_time,
                    dataformats="HWC",
                )


def add_new_data_indicator(run_dir: Path) -> None:
    with (run_dir / ".new_data").open("w") as f:
        f.write("1")


def check_new_data_indicator(run_dir: Path) -> bool:
    indicator_path = run_dir / ".new_data"
    if not indicator_path.exists():
        return False
    with indicator_path.open("r") as f:
        indicator_content = f.read().strip()
    return indicator_content != "0" and indicator_content != ""


def remove_new_data_indicator(run_dir: Path) -> None:
    with (run_dir / ".new_data").open("w") as f:
        f.write("0")
