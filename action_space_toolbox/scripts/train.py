import logging
import random
import subprocess
from pathlib import Path

import gym
import hydra
import numpy as np
import omegaconf
import stable_baselines3.common.logger
import torch
from stable_baselines3.common.callbacks import CheckpointCallback

from action_space_toolbox.util.fix_ep_info_buffer_callback import (
    FixEpInfoBufferCallback,
)
from action_space_toolbox.util.sb3_custom_logger import SB3CustomLogger

logger = logging.getLogger(__name__)


def make_env(cfg: omegaconf.DictConfig) -> gym.Env:
    env = gym.make(cfg.env, **cfg.env_args)
    if "action_transformation" in cfg:
        env = hydra.utils.instantiate(cfg.action_transformation, env=env)
    return env


@hydra.main(version_base=None, config_path="conf", config_name="train")
def train(cfg: omegaconf.DictConfig):
    # This needs to be a local import to get the environments registered in action_space_toolbox to work with the hydra
    # joblib launcher (see also https://github.com/facebookresearch/hydra/issues/1802#issuecomment-908722829)
    import action_space_toolbox

    result_commit = subprocess.run(
        ["git", "-C", f"{Path(__file__).parent}", "rev-parse", "HEAD"],
        stdout=subprocess.PIPE,
    )
    logger.debug(
        f"Commit action_space_optimization: {result_commit.stdout.decode().strip()}"
    )

    logger.info(f"Log directory: {Path.cwd()}")

    if cfg.seed is not None:
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        torch.cuda.manual_seed(cfg.seed)

    env = make_env(cfg)

    algorithm = hydra.utils.instantiate(cfg.algorithm.algorithm, env=env)
    tb_output_format = stable_baselines3.common.logger.TensorBoardOutputFormat(
        "tensorboard"
    )
    algorithm.set_logger(
        SB3CustomLogger("tensorboard", [tb_output_format], env.base_env_timestep_factor)
    )
    training_steps = cfg.algorithm.training.steps // env.base_env_timestep_factor
    callbacks = [
        CheckpointCallback(10000, str(Path.cwd() / "checkpoints"), cfg.algorithm.name),
        FixEpInfoBufferCallback(),
    ]
    try:
        algorithm.learn(total_timesteps=training_steps, callback=callbacks)
    finally:
        (Path.cwd() / "done").touch()


if __name__ == "__main__":
    train()
