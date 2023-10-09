import logging
import random
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

import gym
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
from pg_subspaces.callbacks.additional_checkpoints_callback import (
    AdditionalCheckpointsCallback,
)
from pg_subspaces.callbacks.fix_ep_info_buffer_callback import (
    FixEpInfoBufferCallback,
)
from pg_subspaces.metrics.sb3_custom_logger import SB3CustomLogger

logger = logging.getLogger(__name__)


def obj_config_to_type_and_kwargs(conf_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    stable-baselines3' algorithms should not get objects passed to the constructors. Otherwise we need to checkpoint
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


def make_single_env(
    env_cfg: omegaconf.DictConfig,
    action_transformation_cfg: Optional[omegaconf.DictConfig],
    **kwargs,
):
    env = gym.make(env_cfg, **kwargs)
    if action_transformation_cfg is not None:
        env = hydra.utils.instantiate(action_transformation_cfg, env=env)
    # To get the training code working for environments not wrapped with a ControllerBaseWrapper.
    if not hasattr(env, "base_env_timestep_factor"):
        env.base_env_timestep_factor = 1
    return env


def make_vec_env(cfg: omegaconf.DictConfig) -> stable_baselines3.common.vec_env.VecEnv:
    if cfg.algorithm.training.n_envs == 1:
        env = stable_baselines3.common.vec_env.DummyVecEnv(
            [
                lambda: stable_baselines3.common.monitor.Monitor(
                    make_single_env(
                        cfg.env, cfg.get("action_transformation"), **cfg.env_args
                    )
                )
            ]
        )
    else:
        env = stable_baselines3.common.vec_env.SubprocVecEnv(
            [
                lambda: make_single_env(
                    cfg.env, cfg.get("action_transformation"), **cfg.env_args
                )
                for _ in range(cfg.algorithm.training.n_envs)
            ]
        )
        env = stable_baselines3.common.vec_env.VecMonitor(env)
    if "env_wrappers" in cfg.algorithm:
        for wrapper in cfg.algorithm.env_wrappers.values():
            env = hydra.utils.instantiate(wrapper, venv=env)
    return env


@hydra.main(version_base=None, config_path="conf", config_name="train")
def train(cfg: omegaconf.DictConfig) -> None:
    result_commit = subprocess.run(
        ["git", "-C", f"{Path(__file__).parent}", "rev-parse", "HEAD"],
        stdout=subprocess.PIPE,
    )
    logger.debug(
        f"Commit {Path(__file__).parents[2].name}: {result_commit.stdout.decode().strip()}"
    )

    logger.info(f"Log directory: {Path.cwd()}")

    env = make_vec_env(cfg)

    if cfg.seed is not None:
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        torch.cuda.manual_seed(cfg.seed)
        env.seed(cfg.seed)

    algorithm_cfg = obj_config_to_type_and_kwargs(
        omegaconf.OmegaConf.to_container(cfg.algorithm.algorithm)
    )

    algorithm = hydra.utils.instantiate(algorithm_cfg, env=env, _convert_="partial")
    tb_output_format = stable_baselines3.common.logger.TensorBoardOutputFormat(
        "tensorboard"
    )
    base_env_timestep_factor = env.get_attr("base_env_timestep_factor")[0]
    algorithm.set_logger(
        SB3CustomLogger(
            "tensorboard",
            [tb_output_format],
            base_env_timestep_factor,
        )
    )
    checkpoints_path = Path("checkpoints")
    if checkpoints_path.exists():
        checkpoints = [
            int(p.name[len(f"{cfg.algorithm.name}_") : -len("_steps.zip")])
            for p in checkpoints_path.iterdir()
            if p.suffix == ".zip"
        ]
        checkpoint_to_load = max(checkpoints)
        algorithm.load(
            checkpoints_path / f"{cfg.algorithm.name}_{checkpoint_to_load}_steps.zip",
            device=cfg.algorithm.algorithm.device,
        )
        algorithm.num_timesteps = checkpoint_to_load
        if isinstance(
            algorithm, stable_baselines3.common.off_policy_algorithm.OffPolicyAlgorithm
        ):
            algorithm.load_replay_buffer(
                checkpoints_path
                / f"{cfg.algorithm.name}_replay_buffer_{checkpoint_to_load}_steps.pkl",
            )
    else:
        checkpoints_path.mkdir()
        # Save the initial agent
        path = checkpoints_path / f"{cfg.algorithm.name}_0_steps"
        algorithm.save(path)
    eval_env = make_vec_env(cfg)
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
            AdditionalCheckpointsCallback(
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
    training_steps = cfg.algorithm.training.steps // base_env_timestep_factor
    try:
        # Hack to get the evaluation of the initial policy in addition
        eval_callback.init_callback(algorithm)
        eval_callback._on_step()
        algorithm.learn(
            total_timesteps=training_steps - algorithm.num_timesteps,
            callback=callbacks,
            progress_bar=cfg.show_progress,
            reset_num_timesteps=False,
        )
    finally:
        (Path.cwd() / "done").touch()


if __name__ == "__main__":
    train()