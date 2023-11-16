import itertools
from typing import (
    Callable,
    Iterable,
    List,
    Literal,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import gym
import stable_baselines3
import stable_baselines3.common.buffers
import stable_baselines3.common.vec_env
import torch

from pg_subspaces.sb3_utils.common.agent_spec import AgentSpec
from pg_subspaces.sb3_utils.ppo.ppo_loss import ppo_loss

TWrapper = TypeVar("TWrapper", bound=gym.Wrapper)


def maybe_create_agent(
    agent_or_spec: Union[AgentSpec, stable_baselines3.ppo.PPO],
    env: Optional[gym.Env] = None,
) -> stable_baselines3.ppo.PPO:
    if isinstance(agent_or_spec, AgentSpec):
        return agent_or_spec.create_agent(env)
    else:
        return agent_or_spec


def maybe_create_env(
    env_or_factory: Union[
        stable_baselines3.common.vec_env.VecEnv,
        Callable[[], stable_baselines3.common.vec_env.VecEnv],
    ],
) -> stable_baselines3.common.vec_env.VecEnv:
    if isinstance(env_or_factory, stable_baselines3.common.vec_env.VecEnv):
        return env_or_factory
    else:
        return env_or_factory()


def check_wrapped(env: gym.Env, wrapper: Type[TWrapper]) -> bool:
    """
    Checks whether a given environment is wrapped with a given wrapper.

    :param env:     The gym environment to check
    :param wrapper: The wrapper to check for
    :return:        True iff the environment is wrapped in the given wrapper
    """
    while isinstance(env, gym.Wrapper):
        if isinstance(env, wrapper):
            return True
        env = env.env
    return False


def get_space_shape(space: gym.spaces.Space) -> Tuple[int]:
    if isinstance(space, gym.spaces.Discrete):
        return (1,)
    elif isinstance(space, gym.spaces.Box):
        return space.shape
    else:
        raise ValueError(f"Unknown space {type(space)}.")


# TODO: This should not be specific to PPO
def sample_update_trajectory(
    agent_spec: AgentSpec,
    rollout_buffer: stable_baselines3.common.buffers.RolloutBuffer,
    batch_size: Optional[int],
    max_num_steps: Optional[int] = None,
    n_epochs: Union[int, Literal["inf"]] = 1,
    alternative_optimizer_factory: Optional[
        Callable[[Iterable[torch.nn.Parameter]], torch.optim.Optimizer]
    ] = None,
) -> List[List[torch.Tensor]]:
    agent = agent_spec.create_agent()
    if alternative_optimizer_factory is not None:
        optimizer = alternative_optimizer_factory(agent.policy.parameters())
    else:
        optimizer = agent.policy.optimizer
    parameters = []
    assert max_num_steps is not None or n_epochs != "inf"
    epochs_iter = range(n_epochs) if n_epochs != "inf" else itertools.count()
    step = 0
    for _ in epochs_iter:
        for batch in rollout_buffer.get(batch_size):
            loss, _, _, _ = ppo_loss(agent, batch)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                agent.policy.parameters(), agent.max_grad_norm
            )
            optimizer.step()
            parameters.append([p.detach().clone() for p in agent.policy.parameters()])
            step += 1
            if max_num_steps is not None and step >= max_num_steps:
                return parameters
    return parameters
