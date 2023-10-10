from pathlib import Path
from typing import Sequence

from stable_baselines3.common.callbacks import CheckpointCallback

from pg_subspaces.sb3_utils.common.replay_buffer_diff_checkpointer import (
    ReplayBufferDiffCheckpointer,
)


class CustomCheckpointCallback(CheckpointCallback):
    """
    Extension of stable_baselines3's CheckpointCallback that allows to save the model at specified additional steps
    (irrespective of the save frequency).

    :param save_freq:                   Save checkpoints every ``save_freq`` call of the callback.
    :param additional_checkpoints:      A sequence of additional steps at which to save checkpoints
    :param save_path:                   Path to the folder where the model will be saved.
    :param name_prefix:                 Common prefix to the saved models
    :param save_replay_buffer:          Save the model replay buffer
    :param save_vecnormalize:           Save the ``VecNormalize`` statistics
    :param verbose:                     Verbosity level: 0 for no output, 2 for indicating when saving model checkpoint
    """

    def __init__(
        self,
        save_freq: int,
        additional_checkpoints: Sequence,
        save_path: str,
        name_prefix: str = "rl_model",
        save_replay_buffer: bool = False,
        save_vecnormalize: bool = False,
        verbose: int = 0,
    ):
        super().__init__(
            save_freq,
            save_path,
            name_prefix,
            save_replay_buffer,
            save_vecnormalize,
            verbose,
        )
        self.additional_checkpoints = additional_checkpoints
        self.replay_buffer_checkpointer = None

    def _init_callback(self) -> None:
        super()._init_callback()
        self.replay_buffer_checkpointer = ReplayBufferDiffCheckpointer(
            self.model, self.name_prefix, Path(self.save_path)
        )

    def _on_step(self) -> bool:
        if (
            self.n_calls % self.save_freq == 0
            or self.n_calls in self.additional_checkpoints
        ):
            model_path = self._checkpoint_path(extension="zip")
            self.model.save(model_path)
            if self.verbose >= 2:
                print(f"Saving model checkpoint to {model_path}")

            if (
                self.save_replay_buffer
                and hasattr(self.model, "replay_buffer")
                and self.model.replay_buffer is not None
            ):
                # If model has a replay buffer, save it too
                self.replay_buffer_checkpointer.save()
                if self.verbose > 1:
                    print(f"Saving model replay buffer checkpoint")

            if (
                self.save_vecnormalize
                and self.model.get_vec_normalize_env() is not None
            ):
                # Save the VecNormalize statistics
                vec_normalize_path = self._checkpoint_path(
                    "vecnormalize_", extension="pkl"
                )
                self.model.get_vec_normalize_env().save(vec_normalize_path)
                if self.verbose >= 2:
                    print(f"Saving model VecNormalize to {vec_normalize_path}")

        return True
