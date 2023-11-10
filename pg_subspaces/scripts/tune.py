import copy
import logging
from pathlib import Path

import hydra
import numpy as np
import omegaconf
from joblib import delayed, Parallel

from pg_subspaces.metrics.tensorboard_logs import create_event_accumulators, read_scalar
from pg_subspaces.scripts.train import train

logger = logging.getLogger(__name__)


def worker(cfg: omegaconf.DictConfig, seed: int, working_directory: Path) -> float:
    cfg = copy.deepcopy(cfg)
    cfg.seed = seed
    if isinstance(cfg.algorithm.algorithm.policy_kwargs.net_arch, int):
        # Hack: The sweeper cannot handle list-type parameters, so pass a single number and convert it to a list
        cfg.algorithm.algorithm.policy_kwargs.net_arch = omegaconf.ListConfig(
            [cfg.algorithm.algorithm.policy_kwargs.net_arch]
        )
    working_directory.mkdir()
    try:
        train(cfg, root_path=str(working_directory))
    except Exception as e:
        logger.warning(f"Run {working_directory} failed with exception: {e}")
        return np.inf
    _, event_accumulator = create_event_accumulators(
        [working_directory / "tensorboard"]
    )[0]
    scalar = read_scalar(event_accumulator, "eval/mean_reward")
    if len(scalar) == 1:
        logger.warning(
            f"Logs at {working_directory / 'tensorboard'} contain only a single value."
        )
    return -sum(
        v.value for s, v in scalar.items() if 0 < s < cfg.algorithm.training.steps
    )


@hydra.main(version_base=None, config_path="conf", config_name="tune")
def tune_hyperparams(cfg: omegaconf.DictConfig):
    results = Parallel(n_jobs=cfg.num_seeds, backend="sequential")(
        # Need to pass the working directory here since otherwise loky messes up the paths for some reason
        delayed(worker)(cfg, seed, Path.cwd().absolute() / str(seed))
        for seed in range(cfg.num_seeds)
    )

    return np.mean(results)


if __name__ == "__main__":
    omegaconf.OmegaConf.register_new_resolver("ADD", lambda x, y: x + y)
    omegaconf.OmegaConf.register_new_resolver("SUB", lambda x, y: x - y)
    omegaconf.OmegaConf.register_new_resolver("MUL", lambda x, y: x * y)
    omegaconf.OmegaConf.register_new_resolver("DIV", lambda x, y: x / y)
    omegaconf.OmegaConf.register_new_resolver("INTDIV", lambda x, y: x // y)

    tune_hyperparams()
