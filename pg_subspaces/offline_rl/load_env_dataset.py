from pathlib import Path
from typing import Union, Tuple

import gym
import numpy as np
import omegaconf
import stable_baselines3.common.buffers
import stable_baselines3.common.monitor
import stable_baselines3.common.vec_env
import torch

from pg_subspaces.sb3_utils.common.replay_buffer_diff_checkpointer import (
    get_replay_buffer_checkpoints,
    load_replay_buffer,
)
from pg_subspaces.scripts.train import make_vec_env


def load_env_dataset(
    dataset_spec: str,
    device: Union[str, torch.device],
) -> Tuple[gym.Env, stable_baselines3.common.buffers.ReplayBuffer]:
    if dataset_spec.startswith("d4rl"):
        env_name = dataset_spec[len("d4rl_") :]
        eval_env = stable_baselines3.common.vec_env.DummyVecEnv(
            [lambda: stable_baselines3.common.monitor.Monitor(gym.make(env_name))]
        )
        dataset_path = (
            Path.home() / ".d4rl" / "datasets" / "walker2d_medium_expert-v2.hdf5"
        )
        dataset = eval_env.envs[0].get_dataset(
            h5path=dataset_path if dataset_path.exists() else None
        )
        replay_buffer = stable_baselines3.common.buffers.ReplayBuffer(
            dataset["observations"].shape[0],
            eval_env.observation_space,
            eval_env.action_space,
            device,
        )
        replay_buffer.observations = dataset["observations"][:, None, :]
        replay_buffer.actions = dataset["actions"][:, None, :]
        replay_buffer.rewards = dataset["rewards"][:, None]
        replay_buffer.dones = np.logical_or(dataset["terminals"], dataset["timeouts"])[
            :, None
        ]
        replay_buffer.pos = 0
        replay_buffer.full = True
    else:
        dataset_logs_path = Path(dataset_spec)
        dataset_cfg_path = dataset_logs_path / ".hydra" / "config.yaml"
        dataset_cfg = omegaconf.OmegaConf.load(dataset_cfg_path)
        eval_env = make_vec_env(dataset_cfg)
        name_prefix = "sac"
        checkpoint_dir = dataset_logs_path / "checkpoints"
        rb_checkpoints = get_replay_buffer_checkpoints(
            name_prefix, None, checkpoint_dir
        )
        buffer_size = rb_checkpoints[-1][0]
        replay_buffer = stable_baselines3.common.buffers.ReplayBuffer(
            buffer_size, eval_env.observation_space, eval_env.action_space, device
        )
        load_replay_buffer(
            checkpoint_dir, name_prefix, replay_buffer, None, different_buffer=True
        )
    return eval_env, replay_buffer
