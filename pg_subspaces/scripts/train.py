import logging
import random
import subprocess
import time
import os
from pathlib import Path
from typing import Any, Dict

import hydra
import numpy as np
import omegaconf

import stable_baselines3.common.callbacks
import stable_baselines3.common.logger
import stable_baselines3.common.monitor
import stable_baselines3.common.on_policy_algorithm
import stable_baselines3.common.vec_env
import stable_baselines3.common.off_policy_algorithm
import torch

from pg_subspaces.callbacks.additional_training_metrics_callback import (
    AdditionalTrainingMetricsCallback,
)
from pg_subspaces.callbacks.custom_checkpoint_callback import (
    CustomCheckpointCallback,
)
from pg_subspaces.callbacks.fix_ep_info_buffer_callback import (
    FixEpInfoBufferCallback,
)
from pg_subspaces.metrics.sb3_custom_logger import SB3CustomLogger
from pg_subspaces.sb3_utils.common.agent_spec import CheckpointAgentSpec, HydraAgentSpec
from pg_subspaces.sb3_utils.common.env.make_env import make_vec_env


logger = logging.getLogger(__name__)


def env_with_prefix(key: str, prefix: str, default: str) -> str:
    value = os.getenv(key)
    if value:
        return prefix + value
    return default


def obj_config_to_type_and_kwargs(conf_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    stable-baselines3' algorithms should not get objects passed to the constructors. Otherwise, we need to checkpoint
    the entire object to allow save / load. Therefore, replace each object in the config by a <obj>_type and
    <obj>_kwargs to allow creating them in the algorithm's constructor.

    :param conf_dict:
    :return:
    """
    new_conf = {}
    for key in conf_dict.keys():
        if not isinstance(conf_dict[key], dict):
            new_conf[key] = conf_dict[key]
        elif "_target_" in conf_dict[key] and not key == "activation_fn":
            new_conf[f"{key}_class"] = hydra.utils.get_class(conf_dict[key]["_target_"])
            conf_dict[key].pop("_target_")
            new_conf[f"{key}_kwargs"] = obj_config_to_type_and_kwargs(conf_dict[key])
        elif "_target_" in conf_dict[key] and key == "activation_fn":
            new_conf[key] = hydra.utils.get_class(conf_dict[key]["_target_"])
        else:
            new_conf[key] = obj_config_to_type_and_kwargs(conf_dict[key])
    return new_conf


@hydra.main(version_base=None, config_path="conf", config_name="train")
def main(cfg: omegaconf.DictConfig) -> None:
    train(cfg)


def train(cfg: omegaconf.DictConfig, root_path: str = ".") -> None:
    root_path = Path(root_path)
    result_commit = subprocess.run(
        ["git", "-C", f"{Path(__file__).parent}", "rev-parse", "HEAD"],
        stdout=subprocess.PIPE,
    )
    logger.debug(
        f"Commit {Path(__file__).parents[2].name}: {result_commit.stdout.decode().strip()}"
    )

    logger.info(f"Log directory: {root_path.absolute()}")

    env = make_vec_env(cfg)

    if cfg.seed is not None:
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        torch.cuda.manual_seed(cfg.seed)
        env.seed(cfg.seed)

    omegaconf.OmegaConf.resolve(cfg)
    algorithm_cfg = {
        "algorithm": obj_config_to_type_and_kwargs(
            omegaconf.OmegaConf.to_container(cfg.algorithm.algorithm)
        )
    } | {k: v for k, v in cfg.algorithm.items() if k != "algorithm"}
    checkpoints_path = root_path / "checkpoints"
    # If checkpoints exist, load the checkpoint else train an agent from scratch
    if checkpoints_path.exists():
        checkpoints = [
            int(p.name[len(f"{cfg.algorithm.name}_") : -len("_steps.zip")])
            for p in checkpoints_path.iterdir()
            if p.suffix == ".zip"
        ]
        checkpoint_to_load = max(checkpoints)
        algorithm = CheckpointAgentSpec(
            hydra.utils.get_class(cfg.algorithm.algorithm._target_),
            checkpoints_path,
            checkpoint_to_load,
            cfg.algorithm.algorithm.device,
        ).create_agent(env)
        algorithm.num_timesteps = checkpoint_to_load
    else:
        algorithm = HydraAgentSpec(algorithm_cfg, None, None, None).create_agent(env)
        if cfg.checkpoint_interval is not None:
            checkpoints_path.mkdir()
            # Save the initial agent
            path = checkpoints_path / f"{cfg.algorithm.name}_0_steps"
            algorithm.save(path)
    tb_output_format = stable_baselines3.common.logger.TensorBoardOutputFormat(
        str(root_path / "tensorboard")
    )
    algorithm.set_logger(
        SB3CustomLogger(
            str(root_path / "tensorboard"),
            [tb_output_format],
        )
    )
    eval_env = make_vec_env(cfg)
    if cfg.seed is not None:
        eval_env.seed(cfg.seed + 1)  # Use a different seed for the eval environment
    eval_callback = stable_baselines3.common.callbacks.EvalCallback(
        eval_env,
        n_eval_episodes=cfg.num_eval_episodes,
        eval_freq=cfg.eval_interval,
        verbose=0,
    )
    callbacks = [
        FixEpInfoBufferCallback(),
        eval_callback,
    ]
    if cfg.checkpoint_interval is not None:
        callbacks.append(
            CustomCheckpointCallback(
                cfg.checkpoint_interval,
                cfg.additional_checkpoints,
                str(checkpoints_path),
                cfg.algorithm.name,
                save_replay_buffer=cfg.save_replay_buffer,
                save_vecnormalize=True,
            )
        )
    if isinstance(
        algorithm, stable_baselines3.common.on_policy_algorithm.OnPolicyAlgorithm
    ):
        callbacks.append(AdditionalTrainingMetricsCallback())
    try:
        # Hack to get the evaluation of the initial policy in addition
        eval_callback.init_callback(algorithm)
        eval_callback._on_step()
        algorithm.learn(
            total_timesteps=cfg.algorithm.training.steps - algorithm.num_timesteps,
            callback=callbacks,
            progress_bar=cfg.show_progress,
            reset_num_timesteps=False,
        )
        time.sleep(1)  # To give the tensorboard loggers time to finish writing
    finally:
        (root_path / "done").touch()


if __name__ == "__main__":
    omegaconf.OmegaConf.register_new_resolver("ADD", lambda x, y: x + y)
    omegaconf.OmegaConf.register_new_resolver("SUB", lambda x, y: x - y)
    omegaconf.OmegaConf.register_new_resolver("MUL", lambda x, y: x * y)
    omegaconf.OmegaConf.register_new_resolver("DIV", lambda x, y: x / y)
    omegaconf.OmegaConf.register_new_resolver("INTDIV", lambda x, y: x // y)
    omegaconf.OmegaConf.register_new_resolver("env_with_prefix", env_with_prefix)

    main()
