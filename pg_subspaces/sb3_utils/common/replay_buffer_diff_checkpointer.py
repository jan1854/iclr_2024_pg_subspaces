import re
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import stable_baselines3.common.off_policy_algorithm


class ReplayBufferDiffCheckpointer:
    def __init__(
        self,
        algorithm: stable_baselines3.common.off_policy_algorithm.OffPolicyAlgorithm,
        name_prefix: str,
        checkpoint_dir: Path,
    ):
        self.algorithm = algorithm
        assert not self.algorithm.optimize_memory_usage
        self.name_prefix = name_prefix
        self.checkpoint_dir = checkpoint_dir

    def save(self) -> None:
        if self.algorithm.num_timesteps > 0:
            rb = self.algorithm.replay_buffer
            prev_checkpoints = self._get_replay_buffer_checkpoints(None)
            # The first checkpoint
            if (
                len(prev_checkpoints) == 0
                or len(prev_checkpoints) == 1
                and prev_checkpoints[0][0] == self.algorithm.num_timesteps
            ):
                last_stored_sample = 0
            # In case there already exists a checkpoint for the current timestep
            elif prev_checkpoints[-1][0] == self.algorithm.num_timesteps:
                # The replay buffer is always saved before the current step is added to the buffer
                #   --> At timestep t, the buffer contains t - 1 steps
                last_stored_sample = prev_checkpoints[-2][0] - 1
            else:
                last_stored_sample = prev_checkpoints[-1][0] - 1
            assert (
                self.algorithm.num_timesteps - 1
            ) - last_stored_sample <= rb.buffer_size
            first_pos_to_save = (
                last_stored_sample % self.algorithm.replay_buffer.buffer_size
            )

            fields = [
                (rb.observations, "observations"),
                (rb.actions, "actions"),
                (rb.rewards, "rewards"),
                (rb.dones, "dones"),
                (rb.next_observations, "next_observations"),
            ]

            data = {}
            for field, name in fields:
                if rb.pos > first_pos_to_save:
                    data[name] = field[first_pos_to_save : rb.pos]
                else:
                    data[name] = np.concatenate(
                        (field[first_pos_to_save : rb.buffer_size], field[: rb.pos])
                    )

            data["pos"] = rb.pos
            data["full"] = rb.full

            np.savez_compressed(
                str(
                    self.checkpoint_dir
                    / self._replay_buffer_checkpoint_name(self.algorithm.num_timesteps)
                ),
                **data,
            )

    def load(self, timestep: Optional[int] = None) -> None:
        checkpoints = self._get_replay_buffer_checkpoints(timestep)
        if len(checkpoints) == 0 or checkpoints[-1][0] != timestep:
            raise FileNotFoundError(
                f"Did not find a replay buffer checkpoint for step {timestep}."
            )
        self.algorithm.replay_buffer.pos = 0
        self.algorithm.replay_buffer.full = False
        # TODO: Inefficient, we might avoid reading earlier checkpoints to the buffer if they are overwritten anyway
        for checkpoint in checkpoints:
            data = np.load(checkpoint[1], allow_pickle=True)
            assert (
                self.algorithm.replay_buffer.pos + data["observations"].shape[0]
            ) % self.algorithm.replay_buffer.buffer_size == data["pos"]
            self._add_samples_to_buffer(
                data["observations"],
                data["actions"],
                data["rewards"],
                data["dones"],
                data["next_observations"],
            )
            self.algorithm.replay_buffer.pos = data["pos"]
            self.algorithm.replay_buffer.full = data["full"]

    def _add_samples_to_buffer(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
        next_observations: np.ndarray,
    ) -> None:
        rb = self.algorithm.replay_buffer
        assert rb.pos + len(observations) < 2 * rb.buffer_size
        num_samples_buffer_start = max(rb.pos + len(observations) - rb.buffer_size, 0)
        num_samples_buffer_end = len(observations) - num_samples_buffer_start
        fields = [
            (rb.observations, observations),
            (rb.actions, actions),
            (rb.rewards, rewards),
            (rb.dones, dones),
            (rb.next_observations, next_observations),
        ]
        for rb_field, data_field in fields:
            rb_field[rb.pos : rb.pos + num_samples_buffer_end] = data_field[
                :num_samples_buffer_end
            ]
            rb_field[:num_samples_buffer_start] = data_field[num_samples_buffer_end:]
        rb.full = rb.full or rb.pos + len(observations) >= rb.buffer_size
        rb.pos = (rb.pos + len(observations)) % rb.buffer_size

    def _replay_buffer_checkpoint_name(self, num_timesteps: Union[str, int]) -> str:
        return f"{self.name_prefix}_replay_buffer_{num_timesteps}_steps_diff.npz"

    def _get_replay_buffer_checkpoints(
        self, step: Optional[int]
    ) -> List[Tuple[int, Path]]:
        if step is None:
            step = self.algorithm.num_timesteps
        checkpoints = self.checkpoint_dir.glob(self._replay_buffer_checkpoint_name("*"))
        steps_checkpoints = [
            (int(re.search("_[0-9]+_", c.name).group()[1:-1]), c) for c in checkpoints
        ]
        steps_checkpoints.sort(key=lambda x: x[0])
        return [(s, c) for s, c in steps_checkpoints if s <= step]
