from typing import Any

import torch
import torch.nn.functional as func
from torch import nn


def max_neg_value(t) -> float:
    """Return the maximum negative value for a given tensor type."""
    return -torch.finfo(t.dtype).max


def default(val: Any, d: Any) -> Any:
    """Return the default value if the given value does not exist."""
    return val if (val is not None) else d


def l2norm(t: torch.Tensor) -> torch.Tensor:
    """Apply L2 normalization to the last dimension of a tensor."""
    return func.normalize(t, dim=-1)


def small_init_(t: nn.Linear) -> None:
    """Initialize a linear layer with small weights and zero bias."""
    nn.init.normal_(t.weight, std=0.02)
    nn.init.zeros_(t.bias)
