from pathlib import Path
from typing import Optional

import gym
import hydra
import omegaconf
import stable_baselines3.common.monitor
import stable_baselines3.common.vec_env


def make_single_env(
    env_cfg: omegaconf.DictConfig,
    action_transformation_cfg: Optional[omegaconf.DictConfig],
    **kwargs,
):
    env = gym.make(env_cfg, **kwargs)
    if not isinstance(env.observation_space, gym.spaces.Box):
        env = gym.wrappers.FlattenObservation(env)
    if action_transformation_cfg is not None:
        env = hydra.utils.instantiate(action_transformation_cfg, env=env)
    return env


def make_vec_env(
    cfg: omegaconf.DictConfig,
    vec_normalize_load_path: Optional[Path] = None,
    freeze_loaded_vec_normalize: bool = False,
) -> stable_baselines3.common.vec_env.VecEnv:
    if "n_envs" not in cfg.algorithm.training or cfg.algorithm.training.n_envs == 1:
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
    if vec_normalize_load_path is not None:
        # TODO: This cannot deal with the case that VecNormalize is not the outer-most wrapper
        assert isinstance(env, stable_baselines3.common.vec_env.VecNormalize)
        env = stable_baselines3.common.vec_env.VecNormalize.load(
            str(vec_normalize_load_path), env.venv
        )
        env.training = not freeze_loaded_vec_normalize
    return env
