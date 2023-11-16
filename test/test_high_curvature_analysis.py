import tempfile
from math import sqrt
from pathlib import Path

import gym
import pytest
import stable_baselines3.common.buffers
import stable_baselines3.common.vec_env
import torch

from pg_subspaces.analysis.hessian.hessian_eigen_cached_calculator import (
    HessianEigenCachedCalculator,
)
from pg_subspaces.analysis.high_curvature_subspace_analysis.subspace_overlaps import (
    SubspaceOverlaps,
)
from pg_subspaces.sb3_utils.common.buffer import fill_rollout_buffer
from pg_subspaces.sb3_utils.hessian.eigen.hessian_eigen_explicit import (
    HessianEigenExplicit,
)


def test_high_curvature_overlap():
    v1 = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]).T
    overlap = SubspaceOverlaps._calculate_eigenvectors_overlap(v1, v1)
    assert overlap == pytest.approx(1.0)
    v2 = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]).T
    overlap = SubspaceOverlaps._calculate_eigenvectors_overlap(v1, v2)
    assert overlap == pytest.approx(0.5)
    v2 = torch.tensor([[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]).T
    overlap = SubspaceOverlaps._calculate_eigenvectors_overlap(v1, v2)
    assert overlap == pytest.approx(0.0)
    v2 = torch.tensor(
        [[sqrt(0.5), sqrt(0.5), 0.0, 0.0], [0.0, sqrt(0.5), sqrt(0.5), 0.0]]
    ).T
    overlap = SubspaceOverlaps._calculate_eigenvectors_overlap(v1, v2)
    assert overlap == pytest.approx(0.5 * (1.0 + 0.5))


def test_hessian_eigen_cached_calculator():
    env = stable_baselines3.common.vec_env.DummyVecEnv(
        [lambda: gym.make("Pendulum-v1")]
    )
    agent = stable_baselines3.ppo.PPO(
        "MlpPolicy",
        env,
        device="cpu",
        policy_kwargs={"net_arch": {"pi": [2, 6], "vf": [4, 5]}},
    )
    rollout_buffer = stable_baselines3.common.buffers.RolloutBuffer(
        1000, env.observation_space, env.action_space, device="cpu"
    )
    fill_rollout_buffer(env, agent, rollout_buffer)

    with tempfile.TemporaryDirectory() as tmpdir:
        hess_eigen_comp = HessianEigenCachedCalculator(
            Path(tmpdir), HessianEigenExplicit()
        )
        eigenvals_calc, eigenvecs_calc = hess_eigen_comp.get_eigen_combined_loss(
            agent,
            next(rollout_buffer.get()),
            0,
            num_eigenvectors=30,
        )
        assert torch.all(eigenvals_calc[:-1] >= eigenvals_calc[1:])

        eigenvals_cache, eigenvecs_cache = hess_eigen_comp.get_eigen_combined_loss(
            agent,
            next(rollout_buffer.get()),
            0,
            num_eigenvectors=25,
            calculate_if_no_cached_value=False,
        )
        assert eigenvals_cache == pytest.approx(eigenvals_calc, abs=1e-3)
        assert eigenvecs_cache == pytest.approx(eigenvecs_calc[:, :25], abs=1e-3)

        (eigenvals_pol, eigenvecs_pol), (
            eigenvals_vf,
            eigenvecs_vf,
        ) = hess_eigen_comp.get_eigen_policy_vf_loss(
            agent, next(rollout_buffer.get()), 0, num_eigenvectors=30
        )

        assert torch.all(eigenvals_pol[:-1] >= eigenvals_pol[1:])
        assert torch.all(eigenvals_vf[:-1] >= eigenvals_vf[1:])

        # Check that every eigenvector of the combined loss is also an eigenvector of the policy and value function loss
        # and that the corresponding eigenvalues are the same.
        for i in range(eigenvecs_calc.shape[1]):
            indices_pol = torch.argwhere(
                torch.all(
                    eigenvecs_pol.isclose(eigenvecs_calc[:, i].unsqueeze(1)), dim=0
                )
            )
            indices_vf = torch.argwhere(
                torch.all(
                    eigenvecs_vf.isclose(eigenvecs_calc[:, i].unsqueeze(1)), dim=0
                )
            )
            assert len(indices_pol) == 1 or len(indices_vf) == 1
            if len(indices_pol) == 1:
                assert eigenvals_pol[indices_pol.item()] == eigenvals_calc[i]
            if len(indices_vf) == 1:
                assert eigenvals_vf[indices_vf.item()] == eigenvals_calc[i]

            # Check for duplicate eigenvectors
            for evs in [eigenvecs_calc, eigenvecs_pol, eigenvecs_vf]:
                for i in range(evs.shape[1]):
                    assert (
                        torch.sum(torch.all(evs == evs[:, i].unsqueeze(1), dim=0)) == 1
                    )
