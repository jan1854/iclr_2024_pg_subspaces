import collections
import sys
import time
from collections import deque
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import gym.spaces
import numpy as np
import torch
from stable_baselines3.common import utils
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    ConvertCallback,
    ProgressBarCallback,
)
from stable_baselines3.common.logger import Logger
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import (
    Schedule,
    MaybeCallback,
    ReplayBufferSamples,
)
from stable_baselines3.common.utils import (
    get_schedule_fn,
    set_random_seed,
    get_device,
    update_learning_rate,
    safe_mean,
)
from stable_baselines3.sac.policies import SACPolicy


class DummyEnv:
    num_envs = 1


class OfflineAlgorithm:
    policy_aliases: Dict[str, Type[BasePolicy]] = {}

    def __init__(
        self,
        policy: Union[str, Type[SACPolicy]],
        replay_buffer: ReplayBuffer,
        learning_rate: Union[float, Schedule] = 3e-4,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        action_noise: Optional[ActionNoise] = None,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        stats_window_size: int = 100,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[torch.device, str] = "auto",
        _init_setup_model: bool = True,
    ):
        if isinstance(policy, str):
            self.policy_class = self._get_policy_from_name(policy)
        else:
            self.policy_class = policy

        self.replay_buffer = replay_buffer

        self.device = get_device(device)
        if verbose > 0:
            print(f"Using {self.device} device")

        self.verbose = verbose
        self.policy_kwargs = {} if policy_kwargs is None else policy_kwargs
        self.observation_space: Optional[gym.spaces.Space] = None
        self.action_space: Optional[gym.spaces.Space] = None
        self.num_timesteps = 0
        # Used for updating schedules
        self._total_timesteps = 0
        # Used for computing fps, it is updated at each call of learn()
        self._num_timesteps_at_start = 0
        self.seed = seed
        self.start_time = None
        self.policy = None
        self.learning_rate = learning_rate
        self.tensorboard_log = tensorboard_log
        self.lr_schedule = None  # type: Optional[Schedule]
        # Track the training progress remaining (from 1 to 0)
        # this is used to update the learning rate
        self._current_progress_remaining = 1
        # Buffers for logging
        self._stats_window_size = stats_window_size
        # Buffers for logging
        self.ep_info_buffer: Optional[deque] = None
        self.ep_success_buffer: Optional[deque] = None
        # For logging (and TD3 delayed updates)
        self._n_updates: int = 0
        # The logger object
        self._logger: Optional[Logger] = None
        # Whether the user passed a custom logger or not
        self._custom_logger = False

        self.observation_space = self.replay_buffer.observation_space
        self.action_space = self.replay_buffer.action_space

        # Catch common mistake: using MlpPolicy/CnnPolicy instead of MultiInputPolicy
        if policy in ["MlpPolicy", "CnnPolicy"] and isinstance(
            self.observation_space, gym.spaces.Dict
        ):
            raise ValueError(
                f"You must use `MultiInputPolicy` when working with dict observation space, not {policy}"
            )

        if isinstance(self.action_space, gym.spaces.Box):
            assert np.all(
                np.isfinite(np.array([self.action_space.low, self.action_space.high]))
            ), "Continuous action space must have a finite lower and upper bound"

        self.batch_size = batch_size
        self.tau = tau
        self.gamma = gamma
        self.action_noise = action_noise

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 5000,
        tb_log_name: str = "run",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> "OfflineAlgorithm":
        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )

        callback.on_training_start(locals(), globals())

        while self.num_timesteps < total_timesteps:
            for batch in self.sample_epoch_from_buffer():
                if self.num_timesteps >= total_timesteps:
                    break
                callback.on_step()
                self.train(batch)
                self.num_timesteps += 1
                if log_interval is not None and self.num_timesteps % log_interval == 0:
                    self._dump_logs()

        callback.on_training_end()

        return self

    def train(self, batch: ReplayBufferSamples) -> None:
        raise NotImplementedError()

    def set_random_seed(self, seed: Optional[int] = None) -> None:
        """
        Set the seed of the pseudo-random generators
        (python, numpy, pytorch, gym, action_space)

        :param seed:
        """
        if seed is None:
            return
        set_random_seed(seed, using_cuda=self.device.type == torch.device("cuda").type)
        self.action_space.seed(seed)

    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        self.policy = self.policy_class(  # pytype:disable=not-instantiable
            self.observation_space,
            self.action_space,
            self.lr_schedule,
            **self.policy_kwargs,  # pytype:disable=not-instantiable
        )
        self.policy = self.policy.to(self.device)

    def _setup_lr_schedule(self) -> None:
        """Transform to callable if needed."""
        self.lr_schedule = get_schedule_fn(self.learning_rate)

    def _get_policy_from_name(self, policy_name: str) -> Type[BasePolicy]:
        """
        Get a policy class from its name representation.

        The goal here is to standardize policy naming, e.g.
        all algorithms can call upon "MlpPolicy" or "CnnPolicy",
        and they receive respective policies that work for them.

        :param policy_name: Alias of the policy
        :return: A policy class (type)
        """

        if policy_name in self.policy_aliases:
            return self.policy_aliases[policy_name]
        else:
            raise ValueError(f"Policy {policy_name} unknown")

    def _update_learning_rate(
        self, optimizers: Union[List[torch.optim.Optimizer], torch.optim.Optimizer]
    ) -> None:
        """
        Update the optimizers learning rate using the current learning rate schedule
        and the current progress remaining (from 1 to 0).

        :param optimizers:
            An optimizer or a list of optimizers.
        """
        # Log the current learning rate
        self.logger.record(
            "train/learning_rate", self.lr_schedule(self._current_progress_remaining)
        )

        if not isinstance(optimizers, list):
            optimizers = [optimizers]
        for optimizer in optimizers:
            update_learning_rate(
                optimizer, self.lr_schedule(self._current_progress_remaining)
            )

    def _excluded_save_params(self) -> List[str]:
        """
        Returns the names of the parameters that should be excluded from being
        saved by pickling. E.g. replay buffers are skipped by default
        as they take up a lot of space. PyTorch variables should be excluded
        with this so they can be stored with ``th.save``.

        :return: List of parameters that should be excluded from being saved with pickle.
        """
        return [
            "policy",
            "device",
            "replay_buffer",
            "_episode_storage",
            "_logger",
            "_custom_logger",
        ]

    def _setup_learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        reset_num_timesteps: bool = True,
        tb_log_name: str = "run",
        progress_bar: bool = False,
    ) -> Tuple[int, BaseCallback]:

        self.start_time = time.time_ns()

        if self.ep_info_buffer is None or reset_num_timesteps:
            # Initialize buffers if they don't exist, or reinitialize if resetting counters
            self.ep_info_buffer = deque(maxlen=self._stats_window_size)
            self.ep_success_buffer = deque(maxlen=self._stats_window_size)

        if self.action_noise is not None:
            self.action_noise.reset()

        if reset_num_timesteps:
            self.num_timesteps = 0
            self._episode_num = 0
        else:
            # Make sure training timesteps are ahead of the internal counter
            total_timesteps += self.num_timesteps
        self._total_timesteps = total_timesteps
        self._num_timesteps_at_start = self.num_timesteps

        # Configure logger's outputs if no logger was passed
        if not self._custom_logger:
            self._logger = utils.configure_logger(
                self.verbose, self.tensorboard_log, tb_log_name, reset_num_timesteps
            )

        # Create eval callback if needed
        callback = self._init_callback(callback, progress_bar)

        return total_timesteps, callback

    def _init_callback(
        self,
        callback: MaybeCallback,
        progress_bar: bool = False,
    ) -> BaseCallback:
        """
        :param callback: Callback(s) called at every step with state of the algorithm.
        :param progress_bar: Display a progress bar using tqdm and rich.
        :return: A hybrid callback calling `callback` and performing evaluation.
        """
        # Convert a list of callbacks into a callback
        if isinstance(callback, list):
            callback = CallbackList(callback)

        # Convert functional callback to object
        if not isinstance(callback, BaseCallback):
            callback = ConvertCallback(callback)

        # Add progress bar callback
        if progress_bar:
            callback = CallbackList([callback, ProgressBarCallback()])

        callback.init_callback(self)
        return callback

    def _dump_logs(self) -> None:
        """
        Write log.
        """
        time_elapsed = max(
            (time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon
        )
        fps = int((self.num_timesteps - self._num_timesteps_at_start) / time_elapsed)
        # self.logger.record("time/episodes", self._episode_num, exclude="tensorboard")
        if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
            self.logger.record(
                "rollout/ep_rew_mean",
                safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]),
            )
            self.logger.record(
                "rollout/ep_len_mean",
                safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]),
            )
        self.logger.record("time/fps", fps)
        self.logger.record(
            "time/time_elapsed", int(time_elapsed), exclude="tensorboard"
        )
        self.logger.record(
            "time/total_timesteps", self.num_timesteps, exclude="tensorboard"
        )
        if len(self.ep_success_buffer) > 0:
            self.logger.record(
                "rollout/success_rate", safe_mean(self.ep_success_buffer)
            )
        # Pass the number of timesteps for tensorboard
        self.logger.dump(step=self.num_timesteps)

    def sample_epoch_from_buffer(self):
        indices = np.random.permutation(np.arange(self.replay_buffer.buffer_size))
        num_batches = self.replay_buffer.buffer_size // self.batch_size
        indices = indices[: num_batches * self.batch_size].reshape(-1, self.batch_size)

        for indices_batch in indices:
            yield self.replay_buffer._get_samples(indices_batch)

    def set_logger(self, logger: Logger) -> None:
        """
        Setter for for logger object.

        .. warning::

          When passing a custom logger object,
          this will overwrite ``tensorboard_log`` and ``verbose`` settings
          passed to the constructor.
        """
        self._logger = logger
        # User defined logger
        self._custom_logger = True

    def predict(
        self,
        observation: np.ndarray,
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        """
        Get the policy action from an observation (and optional hidden state).
        Includes sugar-coating to handle different observations (e.g. normalizing images).

        :param observation: the input observation
        :param state: The last hidden states (can be None, used in recurrent policies)
        :param episode_start: The last masks (can be None, used in recurrent policies)
            this correspond to beginning of episodes,
            where the hidden states of the RNN must be reset.
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next hidden state
            (used in recurrent policies)
        """
        return self.policy.predict(observation, state, episode_start, deterministic)

    @property
    def logger(self) -> Logger:
        """Getter for the logger object."""
        return self._logger

    def get_env(self):  # TODO: Remove this hack
        return DummyEnv()

    def get_vec_normalize_env(self):  # TODO: Remove this hack
        return None
