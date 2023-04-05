from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union

import gym
import stable_baselines3.common.vec_env
import torch


class AgentSpec:
    def __init__(
        self,
        checkpoint_path: Optional[Path],
        device: Union[str, torch.device],
        override_weights: Optional[Sequence[torch.Tensor]] = None,
        agent_kwargs: Dict[str, Any] = None,
    ):
        if agent_kwargs is not None:
            self.agent_kwargs = agent_kwargs
        else:
            self.agent_kwargs = {}
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.override_weights = override_weights

    def create_agent(
        self,
        env: Optional[Union[gym.Env, stable_baselines3.common.vec_env.VecEnv]] = None,
    ) -> stable_baselines3.ppo.PPO:
        agent = stable_baselines3.ppo.PPO.load(
            self.checkpoint_path, env, self.device, **self.agent_kwargs
        )
        if self.override_weights is not None:
            assert len(list(agent.policy.parameters())) == len(self.override_weights)
            with torch.no_grad():
                for par, w in zip(agent.policy.parameters(), self.override_weights):
                    par.data[:] = w
        return agent

    def copy_with_new_weights(
        self, override_weights: Sequence[torch.Tensor]
    ) -> "AgentSpec":
        return AgentSpec(self.checkpoint_path, self.device, override_weights, self.agent_kwargs)
