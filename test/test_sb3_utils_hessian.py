import gym
import pytest
import stable_baselines3
import stable_baselines3.common.buffers
import torch

from pg_subspaces.sb3_utils.common.buffer import fill_rollout_buffer
from pg_subspaces.sb3_utils.hessian.calculate_hessian import calculate_hessian
from pg_subspaces.sb3_utils.hessian.eigen.hessian_eigen import HessianEigen
from pg_subspaces.sb3_utils.hessian.eigen.hessian_eigen_explicit import (
    HessianEigenExplicit,
)
from pg_subspaces.sb3_utils.hessian.eigen.hessian_eigen_lanczos import (
    HessianEigenLanczos,
)
from pg_subspaces.sb3_utils.ppo.ppo_loss import ppo_loss


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


def compare_approx_eigen_to_explicit(
    hess_eigen_approx: HessianEigen, bound_abs: float, bound_rel: float
) -> None:
    env = gym.make("Pendulum-v1")
    agent = stable_baselines3.ppo.PPO(
        "MlpPolicy",
        env,
        device="cpu",
        policy_kwargs={"net_arch": {"pi": [10, 10], "vf": [10, 15]}},
    )
    rollout_buffer = stable_baselines3.common.buffers.RolloutBuffer(
        1000, env.observation_space, env.action_space, device="cpu"
    )
    fill_rollout_buffer(env, agent, rollout_buffer)
    hess_eigen_explict = HessianEigenExplicit()

    num_evs = 10
    eigen_approx = hess_eigen_approx.calculate_top_eigen(
        agent, next(rollout_buffer.get()), num_evs, eigenvectors_fullsize=False
    )
    eigen_explicit = hess_eigen_explict.calculate_top_eigen(
        agent, next(rollout_buffer.get()), num_evs, eigenvectors_fullsize=False
    )

    for net_eigen_approx, net_eigen_explicit in [
        (eigen_approx.policy, eigen_explicit.policy),
        (eigen_approx.value_function, eigen_explicit.value_function),
    ]:
        for i in range(len(net_eigen_approx.eigenvalues)):
            val_approx = net_eigen_approx.eigenvalues[i]
            vec_approx = net_eigen_approx.eigenvectors[:, i]
            diffs = torch.abs(net_eigen_explicit.eigenvalues - val_approx)
            closest_idx = diffs.argmin()
            assert val_approx == pytest.approx(
                net_eigen_explicit.eigenvalues[closest_idx],
                rel=bound_rel,
                abs=bound_abs,
            )
            assert vec_approx == pytest.approx(
                net_eigen_explicit.eigenvectors[:, closest_idx],
                rel=bound_rel,
                abs=bound_abs,
            ) or -vec_approx == pytest.approx(
                net_eigen_explicit.eigenvectors[:, closest_idx],
                rel=bound_rel,
                abs=bound_abs,
            )
            # The estimated eigenvalues should roughly correspond to the top true eigenvalues but the order does not
            # always seem 100% correct, so give some leeway
            assert closest_idx == pytest.approx(i, abs=2)


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
    hess_eigen_comp = HessianEigenExplicit()
    eigen = hess_eigen_comp.calculate_top_eigen(
        agent, next(rollout_buffer.get()), 0, eigenvectors_fullsize=True
    )
    assert torch.all(eigen.policy.eigenvalues[:-1] >= eigen.policy.eigenvalues[1:])
    assert torch.all(
        eigen.value_function.eigenvalues[:-1] >= eigen.value_function.eigenvalues[1:]
    )
    eigenvals = torch.cat((eigen.policy.eigenvalues, eigen.value_function.eigenvalues))
    eigenvecs = torch.cat(
        (eigen.policy.eigenvectors, eigen.value_function.eigenvectors), dim=1
    )
    for eigenval, eigenvec in zip(eigenvals, eigenvecs.T):
        eigenvec = eigenvec.unsqueeze(1)
        assert eigenval * eigenvec == pytest.approx(hess @ eigenvec, abs=1e-3, rel=1e-3)
    assert torch.norm(eigenvecs, dim=0) == pytest.approx(1.0)

    # Since there is no parameter sharing between policy and value function, each entry is either zero for the
    # policy or value function eigenvectors
    for i in range(eigen.policy.eigenvectors.shape[1]):
        for j in range(eigen.value_function.eigenvectors.shape[1]):
            assert eigen.policy.eigenvectors[:, i] * eigen.value_function.eigenvectors[
                :, j
            ] == pytest.approx(0.0)


def test_compare_lanczos_to_explicit():
    hess_eigen_lanczos = HessianEigenLanczos(1e-5, 10000, None)
    compare_approx_eigen_to_explicit(hess_eigen_lanczos, 1e-4, 1e-5)


def test_hessian_eigen_orthonormal():
    env = gym.make("Pendulum-v1")
    agent = stable_baselines3.ppo.PPO(
        "MlpPolicy",
        env,
        device="cpu",
        policy_kwargs={"net_arch": {"pi": [2, 3], "vf": [2, 3]}},
    )
    rollout_buffer = stable_baselines3.common.buffers.RolloutBuffer(
        1000, env.observation_space, env.action_space, device="cpu"
    )
    fill_rollout_buffer(env, agent, rollout_buffer)
    num_eigenvecs = 10
    hess_eigen = HessianEigenExplicit()
    eigen = hess_eigen.calculate_top_eigen(
        agent, next(rollout_buffer.get()), num_eigenvecs, eigenvectors_fullsize=True
    )
    eigenvecs_pol = eigen.policy.eigenvectors[:, :num_eigenvecs]
    assert eigenvecs_pol.T @ eigenvecs_pol == pytest.approx(
        torch.eye(eigenvecs_pol.shape[1]), abs=1e-5
    )
    eigenvecs_vf = eigen.value_function.eigenvectors[:, :num_eigenvecs]
    assert eigenvecs_vf.T @ eigenvecs_vf == pytest.approx(
        torch.eye(eigenvecs_vf.shape[1]), abs=1e-5
    )
