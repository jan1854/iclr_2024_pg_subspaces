from pathlib import Path

import gym
import pytest
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from action_space_toolbox.analysis.gradient_analysis.gradient_analysis import (
    GradientAnalysis,
)


def env_factory():
    return gym.make("Reacher_VC-v2")


def agent_factory(env):
    return PPO("MlpPolicy", DummyVecEnv([lambda: env]), device="cpu", seed=42)


def test_gradient_batches_averaging():
    gradient_analysis = GradientAnalysis("test", env_factory, agent_factory, Path())

    gradients_batches = torch.rand((23, 42))
    gradients = []
    for i in range(0, 21, 3):
        gradients.append(
            torch.stack(
                (
                    gradients_batches[i],
                    gradients_batches[i + 1],
                    gradients_batches[i + 2],
                )
            ).mean(dim=0)
        )
    gradients_expected = torch.stack(gradients)
    gradients_calc = (
        gradient_analysis.gradient_similarity_analysis.average_gradient_batches(
            gradients_batches, 3
        )
    )
    assert gradients_calc == pytest.approx(gradients_expected)
