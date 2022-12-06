import functools
from typing import Callable, Dict, List, Optional, Tuple, Type, Union, Any

import numpy as np
import torch
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import MlpExtractor
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

from action_space_toolbox.util import metrics
from action_space_toolbox.util.tensorboard_logs import TensorboardLogs


class ValueFunction(torch.nn.Module):
    def __init__(
        self,
        features_dim: int,
        net_arch: List[Union[int, Dict[str, List[int]]]],
        activation_fn: Type[torch.nn.Module],
        ortho_init: bool,
        init_weights: Callable[[torch.nn.Module], None],
        device: torch.device,
    ):
        super().__init__()
        self.mlp_extractor = MlpExtractor(
            features_dim,
            net_arch=net_arch,
            activation_fn=activation_fn,
            device=device,
        )
        self.value_net = torch.nn.Linear(
            self.mlp_extractor.latent_dim_vf, 1, device=device
        )
        # Init weights: use orthogonal initialization
        # with small initial weight for the output
        if ortho_init:
            # TODO: check for features_extractor
            # Values from stable-baselines.
            # features_extractor/mlp values are
            # originally from openai/baselines (default gains/init_scales).
            module_gains = {
                self.mlp_extractor: np.sqrt(2),
                self.value_net: 1,
            }
            for module, gain in module_gains.items():
                module.apply(functools.partial(init_weights, gain=gain))

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        _, latent_vf = self.mlp_extractor(states)
        return self.value_net(latent_vf)


class ValueFunctionTrainer:
    def __init__(
        self,
        batch_size: int,
    ):
        self._gt_value_function_custom_scalars_added = False
        self.batch_size = batch_size

    def fit_value_function(
        self,
        value_function: ValueFunction,
        optimizer: torch.optim.Optimizer,
        states: torch.Tensor,
        values: torch.Tensor,
        epochs: int,
        env_step: int,
        val_data_ratio: float = 0.05,
        show_progress: bool = False,
    ) -> TensorboardLogs:

        train_dataset_size = round((1.0 - val_data_ratio) * states.size(0))
        dataset_train = TensorDataset(
            states[:train_dataset_size], values[:train_dataset_size]
        )
        states_val = states[train_dataset_size:]
        values_val = values[train_dataset_size:]
        dataloader_train = DataLoader(
            dataset_train, self.batch_size, shuffle=True, drop_last=True
        )

        values_train_mean = dataset_train.tensors[1].mean()
        val_loss_baseline, val_metrics_baseline = self._calculate_loss_and_metrics(
            lambda state: values_train_mean.repeat((state.shape[0], 1)),
            states_val,
            values_val,
        )

        logs = TensorboardLogs()

        train_pbar = trange(
            epochs,
            desc="Training the ground truth value function.",
            unit="epoch",
            mininterval=300,
            disable=not show_progress,
        )
        for epoch in train_pbar:
            for states, values in dataloader_train:
                optimizer.zero_grad()
                train_loss, _ = self._calculate_loss_and_metrics(
                    value_function, states, values
                )
                train_loss.backward()
                optimizer.step()
            train_set_losses = []
            with torch.no_grad():
                for states, values in dataloader_train:
                    loss, _ = self._calculate_loss_and_metrics(
                        value_function, states, values
                    )
                    train_set_losses.append(loss.cpu())
                train_set_loss = np.mean(train_set_losses)
                val_loss, val_metrics = self._calculate_loss_and_metrics(
                    value_function, states_val, values_val
                )
            train_pbar.set_description(
                f"Training the ground truth value function (train loss: {train_set_loss:.4f}, "
                f"validation loss: {val_loss.item():.4f}, validation mean absolute error: {val_metrics['mae']:.4f}, "
                f"validation mean relative error: {val_metrics['mre']:.4f})."
            )
            if epochs > 1:
                logs.add_scalar(
                    f"gradient_analysis/zz_details/gt_value_function_train_loss_step_{env_step}",
                    train_set_loss,
                    epoch,
                )
                logs.add_scalar(
                    f"gradient_analysis/zz_details/gt_value_function_val_loss_step_{env_step}",
                    val_loss.item(),
                    epoch,
                )
                logs.add_scalar(
                    f"gradient_analysis/zz_details/gt_value_function_val_mae_step_{env_step}",
                    val_metrics["mae"],
                    epoch,
                )
                logs.add_scalar(
                    f"gradient_analysis/zz_details/gt_value_function_val_mre_step_{env_step}",
                    val_metrics["mre"],
                    epoch,
                )

            logs.add_scalar(
                "gradient_analysis/gt_value_function_train_loss",
                train_set_loss,
                env_step,
            )
            logs.add_scalar(
                "gradient_analysis/gt_value_function_val_loss",
                val_loss.item(),
                env_step,
            )
            logs.add_scalar(
                "gradient_analysis/gt_value_function_val_mae",
                val_metrics["mae"],
                env_step,
            )
            logs.add_scalar(
                "gradient_analysis/gt_value_function_val_mre",
                val_metrics["mre"],
                env_step,
            )
            logs.add_scalar(
                "gradient_analysis/gt_value_function_val_loss_mean_baseline",
                val_loss_baseline.item(),
                env_step,
            )
            logs.add_scalar(
                "gradient_analysis/gt_value_function_val_mae_mean_baseline",
                val_metrics_baseline["mae"],
                env_step,
            )
            logs.add_scalar(
                "gradient_analysis/gt_value_function_val_mre_mean_baseline",
                val_metrics_baseline["mre"],
                env_step,
            )

            layout = {
                "gradient_analysis": {
                    "gt_value_function_loss_train_val": [
                        "Multiline",
                        [
                            "gradient_analysis/gt_value_function_train_loss",
                            "gradient_analysis/gt_value_function_val_loss",
                        ],
                    ],
                    "gt_value_function_loss_model_baseline": [
                        "Multiline",
                        [
                            "gradient_analysis/gt_value_function_val_loss",
                            "gradient_analysis/gt_value_function_val_loss_mean_baseline",
                        ],
                    ],
                }
            }
            logs.add_custom_scalars(layout)
            return logs

    @staticmethod
    def _calculate_loss_and_metrics(
        value_function: Callable[[torch.Tensor], torch.Tensor],
        states: torch.Tensor,
        values: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        values_pred = value_function(states)
        loss = F.mse_loss(values, values_pred)
        mae = torch.mean(torch.abs(values - values_pred)).item()
        mre = metrics.mean_relative_error(values, values_pred)
        return loss, {"mae": mae, "mre": mre}
