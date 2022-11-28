from typing import Sequence

import torch


def weighted_mean(t: Sequence[torch.Tensor], weights: Sequence[float]) -> torch.Tensor:
    t = torch.stack(list(t), dim=0)
    weights = torch.as_tensor(weights, device=t.device).reshape(
        (-1,) + (1,) * (t.ndim - 1)
    )
    return torch.sum(t * weights / torch.sum(weights), dim=0)
