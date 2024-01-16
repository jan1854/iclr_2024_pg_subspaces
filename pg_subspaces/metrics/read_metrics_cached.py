import logging
import re
from pathlib import Path
from typing import Optional, Sequence, List, Tuple

import numpy as np
from tqdm import tqdm

from pg_subspaces.metrics.tensorboard_logs import (
    create_event_accumulators,
    read_scalar,
    check_new_data_indicator,
)

logger = logging.getLogger(__name__)

REGEX_METRICS_TO_CACHE = [
    "eval/mean_reward",
    "rollout/ep_rew_mean",
    "high_curvature_subspace_analysis/.*/gradient_subspace_fraction_1evs/.*_gradient/.*",
    "high_curvature_subspace_analysis/.*/gradient_subspace_fraction_10evs/.*_gradient/.*",
    "high_curvature_subspace_analysis/.*/gradient_subspace_fraction_100evs/.*_gradient/.*",
    "high_curvature_subspace_analysis/.*/overlaps_top1_checkpoint0050000/.*",
    "high_curvature_subspace_analysis/.*/overlaps_top10_checkpoint0050000/.*",
    "high_curvature_subspace_analysis/.*/overlaps_top100_checkpoint0050000/.*",
    "high_curvature_subspace_analysis/.*/overlaps_top1_checkpoint0100000/.*",
    "high_curvature_subspace_analysis/.*/overlaps_top10_checkpoint0100000/.*",
    "high_curvature_subspace_analysis/.*/overlaps_top100_checkpoint0100000/.*",
    "high_curvature_subspace_analysis/.*/overlaps_top1_checkpoint0500000/.*",
    "high_curvature_subspace_analysis/.*/overlaps_top10_checkpoint0500000/.*",
    "high_curvature_subspace_analysis/.*/overlaps_top100_checkpoint0500000/.*",
]
FULL_REGEX_METRICS = re.compile("|".join([f"({r})" for r in REGEX_METRICS_TO_CACHE]))
CACHE_FILE_NAME = "metrics_cache.npz"


def read_metrics_cached(
    log_path: Path,
    keys: Sequence[str],
    verbose: bool = False,
    only_cached: bool = False,
) -> List[Optional[Tuple[np.ndarray, np.ndarray]]]:
    if (log_path / "checkpoints").exists():
        run_dirs = [log_path]
    else:
        run_dirs = sorted(
            [d for d in log_path.iterdir() if d.is_dir() and d.name.isdigit()],
            key=lambda p: int(p.name),
        )
        for run_dir in run_dirs.copy():
            sub_run_dirs = sorted(
                [d for d in run_dir.iterdir() if d.is_dir() and d.name.isdigit()],
                key=lambda p: int(p.name),
            )
            if len(sub_run_dirs) > 0:
                run_dirs.remove(run_dir)
            run_dirs.extend(sub_run_dirs)

    metrics = []
    for run_dir in tqdm(run_dirs, disable=not verbose):
        metric = load_cache(run_dir, keys, only_cached)
        if metric is None and not only_cached:
            metric = read_tensorboard_and_cache_metrics(run_dir, keys)
        if metric is None:
            logger.warning(f"Did not find any of the keys {keys} for run {run_dir}.")
            metrics.append(None)
        else:
            metrics.append((metric[0].astype(int), metric[1]))
    return metrics


def load_cache(
    run_dir: Path, keys: Sequence[str], only_cached: bool
) -> Optional[np.ndarray]:
    cache_path = run_dir / CACHE_FILE_NAME
    if cache_path.exists() and (not check_new_data_indicator(run_dir) or only_cached):
        cache = np.load(cache_path)
        for key in keys:
            if key in cache:
                return cache[key]
    return None


def read_tensorboard_and_cache_metrics(
    run_dir: Path, keys_to_read: Sequence[str]
) -> Optional[np.ndarray]:
    cache = {}
    tb_dir = run_dir / "tensorboard" if (run_dir / "tensorboard").exists() else run_dir
    event_accumulator = create_event_accumulators([tb_dir])[0][1]
    for key in event_accumulator.Tags()["scalars"]:
        if re.fullmatch(FULL_REGEX_METRICS, key):
            scalars = read_scalar(event_accumulator, key)
            cache[key] = np.array(
                sorted(
                    [[s.step, s.value] for s in scalars.values()], key=lambda s: s[0]
                )
            ).T
    np.savez_compressed(str(run_dir / CACHE_FILE_NAME), **cache)

    for key_to_read in keys_to_read:
        if key_to_read in event_accumulator.Tags()["scalars"]:
            scalars = read_scalar(event_accumulator, key_to_read)
            return np.array(
                sorted(
                    [[s.step, s.value] for s in scalars.values()], key=lambda s: s[0]
                )
            ).T
    return None
