import copy
import logging
import os
from pathlib import Path

import hydra
import numpy as np
import omegaconf
from joblib import Parallel, delayed

from pg_subspaces.metrics.tensorboard_logs import create_event_accumulators, read_scalar
from pg_subspaces.scripts.train import train

logger = logging.getLogger(__name__)


def worker(cfg: omegaconf.DictConfig, seed: int, working_directory: Path) -> float:
    cfg = copy.deepcopy(cfg)
    cfg.seed = seed
    # Hack: The sweeper cannot handle list-type parameters, so pass a single number and convert it to a list
    cfg.algorithm.algorithm.policy_kwargs.net_arch = omegaconf.ListConfig(
        [cfg.algorithm.algorithm.policy_kwargs.net_arch]
    )
    orig_dir = Path.cwd()
    working_directory.mkdir()
    try:
        os.chdir(working_directory)
        train(cfg)
    finally:
        os.chdir(orig_dir)
    _, event_accumulator = create_event_accumulators(
        [working_directory / "tensorboard"]
    )[0]
    scalar = read_scalar(event_accumulator, "eval/mean_reward")
    return -sum(v.value for s, v in scalar.items() if s > 0)


@hydra.main(version_base=None, config_path="conf", config_name="tune")
def tune_hyperparams(cfg: omegaconf.DictConfig):
    num_seeds = 5
    seeds = range(num_seeds)
    results = Parallel(n_jobs=num_seeds, backend="sequential")(
        # Need to pass the working directory here since otherwise loky messes up the paths for some reason
        delayed(worker)(cfg, seed, Path.cwd().absolute() / str(seed))
        for seed in seeds
    )

    return np.mean(results)


if __name__ == "__main__":
    tune_hyperparams()
