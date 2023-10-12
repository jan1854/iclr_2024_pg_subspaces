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


def worker(cfg: omegaconf.DictConfig, seed: int) -> float:
    cfg = copy.deepcopy(cfg)
    cfg.seed = seed
    # Hack: The sweeper cannot handle list-type parameters, so pass a single number and convert it to a list
    cfg.algorithm.algorithm.policy_kwargs.net_arch = omegaconf.ListConfig(
        [cfg.algorithm.algorithm.policy_kwargs.net_arch]
    )
    # cfg.run_log_dir = Path(cfg.run_log_dir) / str(seed)
    orig_dir = Path.cwd()
    seed_dir = Path(str(seed))
    seed_dir.mkdir()
    os.chdir(seed_dir)
    train(cfg)
    _, event_accumulator = create_event_accumulators([Path("tensorboard")])[0]
    scalar = read_scalar(event_accumulator, "eval/mean_reward")
    os.chdir(orig_dir)
    return -sum(v.value for s, v in scalar.items() if s > 0)


@hydra.main(version_base=None, config_path="conf", config_name="tune")
def tune_hyperparams(cfg: omegaconf.DictConfig):
    num_seeds = 5
    seeds = range(num_seeds)
    results = Parallel(n_jobs=num_seeds, backend="loky")(
        delayed(worker)(cfg, seed) for seed in seeds
    )

    return np.mean(results)


if __name__ == "__main__":
    tune_hyperparams()
