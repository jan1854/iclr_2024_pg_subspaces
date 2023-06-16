import tempfile
from pathlib import Path

import gym
import pytest
import stable_baselines3
import stable_baselines3.common.buffers
import torch

from action_space_toolbox.analysis.hessian.calculate_hessian import calculate_hessian
from action_space_toolbox.analysis.hessian.hessian_eigen_cached_calculator import (
    HessianEigenCachedCalculator,
)
from action_space_toolbox.util.sb3_training import ppo_loss, fill_rollout_buffer


class DummyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin2 = torch.nn.Linear(2, 2, bias=False)
        self.lin1 = torch.nn.Linear(3, 2, bias=False)

    def forward(self, x):
        return self.lin2(self.lin1(x))


class DummyAgent:
    def __init__(self, policy: DummyPolicy):
        self.policy = policy


def analytic_hessian(x):
    return torch.tensor(
        [
            [0, 0, 0, 0, 2 * x[0], 2 * x[1], 2 * x[2], 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 2 * x[0], 2 * x[1], 2 * x[2]],
            [0, 0, 0, 0, 3 * x[0], 3 * x[1], 3 * x[2], 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 3 * x[0], 3 * x[1], 3 * x[2]],
            [2 * x[0], 0, 3 * x[0], 0, 0, 0, 0, 0, 0, 0],
            [2 * x[1], 0, 3 * x[1], 0, 0, 0, 0, 0, 0, 0],
            [2 * x[2], 0, 3 * x[2], 0, 0, 0, 0, 0, 0, 0],
            [0, 2 * x[0], 0, 3 * x[0], 0, 0, 0, 0, 0, 0],
            [0, 2 * x[1], 0, 3 * x[1], 0, 0, 0, 0, 0, 0],
            [0, 2 * x[2], 0, 3 * x[2], 0, 0, 0, 0, 0, 0],
        ],
        dtype=torch.float32,
    )


def test_calculate_hessian_analytic():
    x = torch.tensor([11, 12, 13], dtype=torch.float32)

    def dummy_loss(agent: DummyAgent) -> torch.Tensor:
        out = agent.policy(x.unsqueeze(0))
        return 2 * out[0, 0] + 3 * out[0, 1]

    dummy_agent = DummyAgent(DummyPolicy())
    parameters = list(dummy_agent.policy.parameters())
    parameters[0].data = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
    parameters[1].data = torch.tensor([[5, 6, 7], [8, 9, 10]], dtype=torch.float32)
    hess = calculate_hessian(dummy_agent, dummy_loss)
    analytic_hess = analytic_hessian(x)
    assert hess == pytest.approx(analytic_hess)


def test_calculate_hessian_dimension():
    env = gym.make("Pendulum-v1")
    agent = stable_baselines3.ppo.PPO(
        "MlpPolicy", env, device="cpu", policy_kwargs={"net_arch": [32, 16]}
    )
    rollout_buffer = stable_baselines3.common.buffers.RolloutBuffer(
        1000, env.observation_space, env.action_space, device="cpu"
    )
    fill_rollout_buffer(env, agent, rollout_buffer)
    hess = calculate_hessian(
        agent, lambda a: ppo_loss(a, next(rollout_buffer.get()))[0]
    )
    num_parameters = len(torch.cat([p.flatten() for p in agent.policy.parameters()]))
    assert hess.shape == (num_parameters, num_parameters)


def test_hessian_ev_calculation():
    env = gym.make("Pendulum-v1")
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
    parameter_names = [n for n, _ in agent.policy.named_parameters()]
    hess = calculate_hessian(
        agent, lambda a: ppo_loss(a, next(rollout_buffer.get()))[0], parameter_names
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        hess_eigen_comp = HessianEigenCachedCalculator(Path(tmpdir))
        eigenvals_calc, eigenvecs_calc = hess_eigen_comp.get_eigen_combined_loss(
            agent, next(rollout_buffer.get()), 0, num_eigenvectors=30
        )
        assert torch.all(eigenvals_calc[:-1] >= eigenvals_calc[1:])
        for eigenval, eigenvec in zip(eigenvals_calc, eigenvecs_calc.T):
            eigenvec = eigenvec.unsqueeze(1)
            assert eigenval * eigenvec == pytest.approx(
                hess @ eigenvec, abs=1e-3, rel=1e-3
            )
            assert torch.norm(eigenvec) == pytest.approx(1.0)
        eigenvals_cache, eigenvecs_cache = hess_eigen_comp.get_eigen_combined_loss(
            agent, next(rollout_buffer.get()), 0, num_eigenvectors=25
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

        # Since there is no parameter sharing between policy and value function, each entry is either zero for the
        # policy or value function eigenvectors
        for i in range(eigenvecs_pol.shape[1]):
            for j in range(eigenvecs_vf.shape[1]):
                assert eigenvecs_pol[:, i] * eigenvecs_vf[:, j] == pytest.approx(0.0)

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
                assert torch.sum(torch.all(evs == evs[:, i].unsqueeze(1), dim=0)) == 1
