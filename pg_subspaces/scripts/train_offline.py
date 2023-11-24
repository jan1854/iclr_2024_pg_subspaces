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

from pg_subspaces.callbacks.custom_checkpoint_callback import (
    CustomCheckpointCallback,
)
from pg_subspaces.callbacks.fix_ep_info_buffer_callback import (
    FixEpInfoBufferCallback,
)
from pg_subspaces.metrics.sb3_custom_logger import SB3CustomLogger
from pg_subspaces.offline_rl.fixed_normalization_wrapper import (
    normalize,
    FixedNormalizationVecWrapper,
)
from pg_subspaces.offline_rl.load_env_dataset import load_env_dataset
from pg_subspaces.utils.hydra import register_custom_resolvers

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


@hydra.main(version_base=None, config_path="conf", config_name="train_offline")
def main(cfg: omegaconf.DictConfig) -> None:
    train_offline(cfg)


def train_offline(cfg: omegaconf.DictConfig, root_path: str = ".") -> None:
    # Registers the environments with datasets. This needs to be here since otherwise the environments are for some
    # reason not registered when running multiple runs concurrently with hydra-joblib-launcher.
    import d4rl

    root_path = Path(root_path)
    result_commit = subprocess.run(
        ["git", "-C", f"{Path(__file__).parent}", "rev-parse", "HEAD"],
        stdout=subprocess.PIPE,
    )
    logger.debug(
        f"Commit {Path(__file__).parents[2].name}: {result_commit.stdout.decode().strip()}"
    )
    logger.info(f"Log directory: {root_path.absolute()}")

    eval_env, replay_buffer = load_env_dataset(cfg.logs_dataset, cfg.device)

    episode_returns_dataset = [0.0]
    for r, done in zip(replay_buffer.rewards, replay_buffer.dones):
        episode_returns_dataset[-1] += r.item()
        if done:
            episode_returns_dataset.append(0.0)
    print(f"Average dataset returns: {np.mean(episode_returns_dataset)}")

    if cfg.seed is not None:
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        torch.cuda.manual_seed(cfg.seed)
        eval_env.seed(cfg.seed)

    omegaconf.OmegaConf.resolve(cfg)
    algorithm_cfg = {
        "algorithm": obj_config_to_type_and_kwargs(
            omegaconf.OmegaConf.to_container(cfg.algorithm.algorithm)
        )
    } | {k: v for k, v in cfg.algorithm.items() if k != "algorithm"}
    algorithm_cfg = omegaconf.DictConfig(algorithm_cfg)
    checkpoints_path = root_path / "checkpoints"
    # If checkpoints exist, load the checkpoint else train an agent from scratch
    if checkpoints_path.exists():
        raise NotImplementedError(
            "Checkpoint loading is not yet implemented for offline algorithms"
        )
        # checkpoints = [
        #     int(p.name[len(f"{cfg.algorithm.name}_") : -len("_steps.zip")])
        #     for p in checkpoints_path.iterdir()
        #     if p.suffix == ".zip"
        # ]
        # checkpoint_to_load = max(checkpoints)
        # algorithm = CheckpointAgentSpec(
        #     hydra.utils.get_class(cfg.algorithm.algorithm._target_),
        #     checkpoints_path,
        #     checkpoint_to_load,
        #     cfg.algorithm.algorithm.device,
        # ).create_agent(env)
        # algorithm.num_timesteps = checkpoint_to_load
    else:
        algorithm = hydra.utils.instantiate(
            algorithm_cfg.algorithm, dataset=replay_buffer, _convert_="partial"
        )
        if cfg.checkpoint_interval is not None:
            checkpoints_path.mkdir()
            # Save the initial agent
            path = checkpoints_path / f"{cfg.algorithm.name}_0_steps"
            algorithm.save(path)

    states_buffer = (
        replay_buffer.observations
        if replay_buffer.full
        else replay_buffer.observations[: replay_buffer.pos]
    )
    states_mean = np.mean(states_buffer, axis=0)
    states_std = np.std(states_buffer, axis=0)
    replay_buffer.observations = normalize(
        replay_buffer.observations,
        states_mean,
        states_std,
        cfg.state_normalization_eps,
    )
    eval_env = FixedNormalizationVecWrapper(
        eval_env, states_mean, states_std, cfg.state_normalization_eps
    )

    tb_output_format = stable_baselines3.common.logger.TensorBoardOutputFormat(
        str(root_path / "tensorboard")
    )
    algorithm.set_logger(
        SB3CustomLogger(
            str(root_path / "tensorboard"),
            [tb_output_format],
        )
    )
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
            )
        )
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
    register_custom_resolvers()

    main()
