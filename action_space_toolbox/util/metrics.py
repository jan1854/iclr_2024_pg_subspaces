import torch


def mean_relative_error(
    ground_truth: torch.Tensor, prediction: torch.Tensor, eps: float = 1e-8
) -> float:
    return torch.mean(
        torch.abs(prediction - ground_truth) / (torch.abs(ground_truth) + eps)
    ).item()
