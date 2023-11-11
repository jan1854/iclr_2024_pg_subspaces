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
