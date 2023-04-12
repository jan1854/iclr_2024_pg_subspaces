from typing import Union

import numpy as np
import torch


def mean_relative_difference(
    original: Union[float, np.ndarray, torch.Tensor],
    modified: Union[np.ndarray, torch.Tensor],
    eps: float = 1e-8,
) -> float:
    return relative_difference(original, modified, eps).mean().item()


def relative_difference(
    original: Union[float, np.ndarray, torch.Tensor],
    modified: Union[float, np.ndarray, torch.Tensor],
    eps: float = 1e-8,
) -> Union[float, np.ndarray, torch.Tensor]:
    return (modified - original) / (abs(original) + eps)


def mean_relative_error(
    ground_truth: Union[np.ndarray, torch.Tensor],
    prediction: Union[np.ndarray, torch.Tensor],
    eps: float = 1e-8,
) -> float:
    return (
        (abs(prediction - ground_truth) / (torch.abs(ground_truth) + eps)).mean().item()
    )
