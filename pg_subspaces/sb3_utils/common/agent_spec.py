import abc
import logging
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence, Union, Type, TypeVar

import gym
import hydra
import omegaconf
import stable_baselines3.common.on_policy_algorithm
import stable_baselines3.common.off_policy_algorithm
import stable_baselines3.common.vec_env
import torch

from pg_subspaces.sb3_utils.common.replay_buffer_diff_checkpointer import (
    ReplayBufferDiffCheckpointer,
)

SB3Agent = TypeVar(
    "SB3Agent", bound=stable_baselines3.common.on_policy_algorithm.OnPolicyAlgorithm
)

logger = logging.getLogger("__name__")


class AgentSpec(abc.ABC):
    def __init__(
        self,
        device: Union[str, torch.device],
        override_weights: Optional[Sequence[torch.Tensor]],
        agent_kwargs: Optional[Dict[str, Any]],
    ):
        self.override_weights = override_weights
        if agent_kwargs is not None:
            self.agent_kwargs = agent_kwargs
        else:
            self.agent_kwargs = {}
        self.device = device

    def create_agent(
        self,
        env: Optional[Union[gym.Env, stable_baselines3.common.vec_env.VecEnv]] = None,
    ) -> SB3Agent:
        agent = self._create_agent(env)
        if self.override_weights is not None:
            self._set_weights(agent, self.override_weights)
        return agent

    @classmethod
    def _set_weights(cls, agent: SB3Agent, weights: Sequence[torch.Tensor]):
        weights = list(weights)
        assert len(list(agent.policy.parameters())) == len(weights)
        with torch.no_grad():
            for par, w in zip(agent.policy.parameters(), weights):
                par.data[:] = w

    @abc.abstractmethod
    def _create_agent(
        self,
        env: Optional[Union[gym.Env, stable_baselines3.common.vec_env.VecEnv]],
    ) -> SB3Agent:
        pass

    @abc.abstractmethod
    def copy_with_new_parameters(
        self,
        weights: Optional[Sequence[torch.Tensor]] = None,
        agent_kwargs: Optional[Dict[str, Any]] = None,
    ) -> "AgentSpec":
        pass


class CheckpointAgentSpec(AgentSpec):
    def __init__(
        self,
        agent_class: Type[SB3Agent],
        checkpoints_dir: Path,
        timestep: int,
        device: Union[str, torch.device],
        override_weights: Optional[Sequence[torch.Tensor]] = None,
        agent_kwargs: Dict[str, Any] = None,
    ):
        super().__init__(device, override_weights, agent_kwargs)
        self.agent_class = agent_class
        self.checkpoints_dir = checkpoints_dir
        self.timestep = timestep

    def _create_agent(
        self,
        env: Optional[Union[gym.Env, stable_baselines3.common.vec_env.VecEnv]],
    ) -> SB3Agent:
        agent_checkpoint_path = list(
            self.checkpoints_dir.glob(f"*_{self.timestep}_steps.zip")
        )
        assert len(agent_checkpoint_path) == 1
        agent_checkpoint_path = agent_checkpoint_path[0]
        agent = self.agent_class.load(
            agent_checkpoint_path, env, self.device, **self.agent_kwargs
        )
        if isinstance(
            agent, stable_baselines3.common.off_policy_algorithm.OffPolicyAlgorithm
        ):
            agent_name = agent_checkpoint_path.name[
                : agent_checkpoint_path.name.find("_")
            ]
            rb_checkpointer = ReplayBufferDiffCheckpointer(
                agent, agent_name, self.checkpoints_dir
            )
            try:
                rb_checkpointer.load(self.timestep)
            except FileNotFoundError:
                replay_buffer_path = (
                    self.checkpoints_dir
                    / f"{agent_name}_replay_buffer_{self.timestep}_steps.pkl"
                )
                agent.load_replay_buffer(replay_buffer_path)
                if replay_buffer_path.exists():
                    agent.load_replay_buffer(replay_buffer_path)
                    agent.replay_buffer.device = self.device
                else:
                    logger.warning(
                        f"Did not find a replay buffer checkpoint for step {self.timestep} "
                        f"in directory {self.checkpoints_dir}."
                    )
        return agent

    def copy_with_new_parameters(
        self,
        weights: Optional[Sequence[torch.Tensor]] = None,
        agent_kwargs: Optional[Dict[str, Any]] = None,
    ) -> "CheckpointAgentSpec":
        if weights is None:
            weights = self.override_weights
        if agent_kwargs is None:
            agent_kwargs = self.agent_kwargs
        return CheckpointAgentSpec(
            self.agent_class,
            self.checkpoints_dir,
            self.timestep,
            self.device,
            weights,
            agent_kwargs,
        )


class HydraAgentSpec(AgentSpec):
    def __init__(
        self,
        agent_cfg: Union[omegaconf.DictConfig, Dict[str, Any]],
        device: Union[str, torch.device, None],
        env_factory: Optional[Callable[[], gym.Env]],
        weights_checkpoint_path: Optional[Path],
        override_weights: Optional[Sequence[torch.Tensor]] = None,
        agent_kwargs: Dict[str, Any] = None,
    ):
        if device is None:
            device = agent_cfg["algorithm"]["device"]
        super().__init__(device, override_weights, agent_kwargs)
        self.agent_cfg = agent_cfg
        self.env_factory = env_factory
        self.weights_checkpoint_path = weights_checkpoint_path

    def _create_agent(
        self,
        env: Optional[Union[gym.Env, stable_baselines3.common.vec_env.VecEnv]],
    ) -> SB3Agent:
        if env is None:
            env = self.env_factory()

        agent = hydra.utils.instantiate(
            self.agent_cfg["algorithm"],
            policy="MlpPolicy",
            env=env,
            device=self.device,
            _convert_="partial",
            **self.agent_kwargs,
        )
        if self.weights_checkpoint_path is not None:
            agent_class = hydra.utils.get_class(self.agent_cfg["algorithm"]["_target_"])
            agent_checkpoint = agent_class.load(
                self.weights_checkpoint_path, env, self.device
            )
            self._set_weights(agent, agent_checkpoint.policy.parameters())

        return agent

    def copy_with_new_parameters(
        self,
        weights: Optional[Sequence[torch.Tensor]] = None,
        agent_kwargs: Optional[Dict[str, Any]] = None,
    ) -> "HydraAgentSpec":
        if weights is None:
            weights = self.override_weights
        if agent_kwargs is None:
            agent_kwargs = self.agent_kwargs
        return HydraAgentSpec(
            self.agent_cfg,
            self.device,
            self.env_factory,
            self.weights_checkpoint_path,
            weights,
            agent_kwargs,
        )
