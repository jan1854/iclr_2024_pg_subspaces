import abc
import logging
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Optional,
    Sequence,
    Union,
    Type,
    TypeVar,
    Literal,
)

import gym
import hydra
import omegaconf
import stable_baselines3.common.buffers
import stable_baselines3.common.on_policy_algorithm
import stable_baselines3.common.off_policy_algorithm
import stable_baselines3.common.vec_env
import torch

from pg_subspaces.offline_rl.offline_algorithm import OfflineAlgorithm
from pg_subspaces.sb3_utils.common.replay_buffer_diff_checkpointer import (
    ReplayBufferDiffCheckpointer,
)

SB3Agent = TypeVar(
    "SB3Agent", bound=stable_baselines3.common.on_policy_algorithm.BaseAlgorithm
)

logger = logging.getLogger(__name__)


def get_agent_name(checkpoints_dir: Path):
    agent_checkpoint_path = next(checkpoints_dir.glob(f"*_steps.zip"))
    return agent_checkpoint_path.name[: agent_checkpoint_path.name.find("_")]


def get_checkpoint_path(
    checkpoints_dir: Path,
    timestep: int,
    checkpoint_type: Optional[Literal["replay_buffer", "vecnormalize"]] = None,
):
    agent_name = get_agent_name(checkpoints_dir)
    checkpoint_type_str = f"{checkpoint_type}_" if checkpoint_type is not None else ""
    checkpoint_path = (
        checkpoints_dir / f"{agent_name}_{checkpoint_type_str}{timestep}_steps"
    )
    if checkpoint_type is None:
        checkpoint_path = checkpoint_path.with_suffix(".zip")
    else:
        checkpoint_path = checkpoint_path.with_suffix(".pkl")
    return checkpoint_path


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
        replay_buffer: Optional[stable_baselines3.common.buffers.ReplayBuffer] = None,
    ) -> SB3Agent:
        agent = self._create_agent(env, replay_buffer)
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
        replay_buffer: Optional[stable_baselines3.common.buffers.ReplayBuffer],
    ) -> SB3Agent:
        pass

    @abc.abstractmethod
    def copy_with_new_parameters(
        self,
        weights: Optional[Sequence[torch.Tensor]] = None,
        agent_kwargs: Optional[Dict[str, Any]] = None,
        device: Optional[Union[torch.device, str]] = None,
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
        freeze_vec_normalize: bool = False,
    ):
        super().__init__(device, override_weights, agent_kwargs)
        self.agent_class = agent_class
        self.checkpoints_dir = checkpoints_dir
        self.timestep = timestep
        self.freeze_vec_normalize = freeze_vec_normalize

    def _create_agent(
        self,
        env: Optional[stable_baselines3.common.vec_env.VecEnv],
        replay_buffer: Optional[stable_baselines3.common.buffers.ReplayBuffer],
    ) -> SB3Agent:
        agent_checkpoint_path = get_checkpoint_path(self.checkpoints_dir, self.timestep)
        if issubclass(self.agent_class, OfflineAlgorithm):
            agent = self.agent_class.load(
                agent_checkpoint_path, replay_buffer, self.device, **self.agent_kwargs
            )
        else:
            agent = self.agent_class.load(
                agent_checkpoint_path, env, self.device, **self.agent_kwargs
            )
            if replay_buffer is not None:
                agent.replay_buffer = replay_buffer
        if replay_buffer is None and isinstance(
            agent, stable_baselines3.common.off_policy_algorithm.OffPolicyAlgorithm
        ):
            agent_name = get_agent_name(self.checkpoints_dir)
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
        device: Optional[Union[torch.device, str]] = None,
    ) -> "CheckpointAgentSpec":
        if weights is None:
            weights = self.override_weights
        if agent_kwargs is None:
            agent_kwargs = self.agent_kwargs
        if device is None:
            device = self.device
        return CheckpointAgentSpec(
            self.agent_class,
            self.checkpoints_dir,
            self.timestep,
            device,
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
        self.agent_cfg = {
            "algorithm": self._obj_config_to_type_and_kwargs(
                omegaconf.OmegaConf.to_container(agent_cfg["algorithm"])
            )
        } | {k: v for k, v in agent_cfg.items() if k != "algorithm"}
        self.env_factory = env_factory
        self.weights_checkpoint_path = weights_checkpoint_path

        if "net_arch" in self.agent_cfg.get("algorithm", {}).get(
            "policy_kwargs", {}
        ) and isinstance(self.agent_cfg["algorithm"]["policy_kwargs"]["net_arch"], int):
            # Hack: The sweeper cannot handle list-type parameters, so if the net_arch is a scalar, convert it to a list
            self.agent_cfg["algorithm"]["policy_kwargs"]["net_arch"] = [
                self.agent_cfg["algorithm"]["policy_kwargs"]["net_arch"],
                self.agent_cfg["algorithm"]["policy_kwargs"]["net_arch"],
            ]

    def _create_agent(
        self,
        env: Optional[stable_baselines3.common.vec_env.VecEnv],
        replay_buffer: Optional[stable_baselines3.common.buffers.ReplayBuffer],
    ) -> SB3Agent:
        agent_class = hydra.utils.get_class(self.agent_cfg["algorithm"]["_target_"])
        if issubclass(agent_class, OfflineAlgorithm):
            agent = hydra.utils.instantiate(
                self.agent_cfg["algorithm"],
                policy="MlpPolicy",
                dataset=replay_buffer,
                device=self.device,
                _convert_="partial",
                **self.agent_kwargs,
            )
        else:
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
            if replay_buffer is not None:
                agent.replay_buffer = replay_buffer

        if self.weights_checkpoint_path is not None:
            agent_checkpoint = agent_class.load(
                self.weights_checkpoint_path, env, self.device
            )
            self._set_weights(agent, agent_checkpoint.policy.parameters())

        return agent

    @classmethod
    def _obj_config_to_type_and_kwargs(cls, cfg_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        stable-baselines3' algorithms should not get objects passed to the constructors. Otherwise, we need to checkpoint
        the entire object to allow save / load. Therefore, replace each object in the config by a <obj>_type and
        <obj>_kwargs to allow creating them in the algorithm's constructor.

        :param cfg_dict:
        :return:
        """
        new_cfg = {}
        for key in cfg_dict.keys():
            if not isinstance(cfg_dict[key], dict):
                new_cfg[key] = cfg_dict[key]
            elif "_target_" in cfg_dict[key] and not key == "activation_fn":
                new_cfg[f"{key}_class"] = hydra.utils.get_class(
                    cfg_dict[key]["_target_"]
                )
                cfg_dict[key].pop("_target_")
                new_cfg[f"{key}_kwargs"] = cls._obj_config_to_type_and_kwargs(
                    cfg_dict[key]
                )
            elif "_target_" in cfg_dict[key] and key == "activation_fn":
                new_cfg[key] = hydra.utils.get_class(cfg_dict[key]["_target_"])
            else:
                new_cfg[key] = cls._obj_config_to_type_and_kwargs(cfg_dict[key])
        return new_cfg

    def copy_with_new_parameters(
        self,
        weights: Optional[Sequence[torch.Tensor]] = None,
        agent_kwargs: Optional[Dict[str, Any]] = None,
        device: Optional[Union[torch.device, str]] = None,
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
