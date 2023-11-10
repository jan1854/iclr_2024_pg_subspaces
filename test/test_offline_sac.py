import tempfile
from pathlib import Path

import d4rl
import numpy as np
import torch

from pg_subspaces.offline_rl.minimalistic_offline_sac import MinimalisticOfflineSAC
from pg_subspaces.scripts.train_offline import load_env_dataset


def test_save_load():
    _, replay_buffer_c = load_env_dataset("d4rl_walker2d-medium-expert-v2", "cpu")
    algorithm_c = MinimalisticOfflineSAC("MlpPolicy", replay_buffer_c, device="cpu")
    with tempfile.TemporaryDirectory() as temp_dir:
        save_path = Path(temp_dir) / "algo.zip"
        algorithm_c.save(save_path)
        _, replay_buffer_l = load_env_dataset("d4rl_walker2d-medium-expert-v2", "cpu")
        algorithm_l = MinimalisticOfflineSAC.load(
            save_path, replay_buffer_l, device="cpu"
        )
    assert np.all(
        [
            torch.all(p_c == p_l)
            for p_c, p_l in zip(
                algorithm_c.policy.parameters(),
                algorithm_l.policy.parameters(),
            )
        ]
    )
    assert algorithm_c.learning_rate == algorithm_l.learning_rate
    assert algorithm_c.batch_size == algorithm_l.batch_size
    assert algorithm_c.tau == algorithm_l.tau
    assert algorithm_c.gamma == algorithm_l.gamma
    assert (
        algorithm_c.actor_rl_objective_weight == algorithm_l.actor_rl_objective_weight
    )
    assert (
        algorithm_c.scale_actor_rl_objective_weight
        == algorithm_l.scale_actor_rl_objective_weight
    )
    assert (
        algorithm_c.actor_bc_objective_weight == algorithm_l.actor_bc_objective_weight
    )
    assert algorithm_c.action_noise == algorithm_l.action_noise
    assert algorithm_c.ent_coef == algorithm_l.ent_coef
    assert algorithm_c.target_update_interval == algorithm_l.target_update_interval
    assert algorithm_c.target_entropy == algorithm_l.target_entropy
    assert algorithm_c.tensorboard_log == algorithm_l.tensorboard_log
    assert algorithm_c.policy_kwargs == algorithm_l.policy_kwargs
    assert algorithm_c.seed == algorithm_l.seed
    assert algorithm_c.device == algorithm_l.device
    assert algorithm_c.policy_kwargs == algorithm_l.policy_kwargs

    assert np.all(
        algorithm_c.replay_buffer.observations == algorithm_l.replay_buffer.observations
    )
    assert np.all(
        algorithm_c.replay_buffer.actions == algorithm_l.replay_buffer.actions
    )
    assert np.all(
        algorithm_c.replay_buffer.rewards == algorithm_l.replay_buffer.rewards
    )
    assert np.all(algorithm_c.replay_buffer.dones == algorithm_l.replay_buffer.dones)
