"""Focal Loss implementation."""

from typing import Literal

import torch
from torch import nn
from torch.nn import functional as func


class FocalLoss(nn.Module):
    """Focal Loss module."""

    def __init__(
        self,
        weight: torch.Tensor | None = None,
        gamma: float = 2.0,
        reduction: Literal["none", "mean", "sum"] = "mean",
    ):
        """Initialize the Focal Loss.

        Args:
            weight (torch.Tensor | None, optional):  a manual rescaling weight given to
                each class. If given, it has to be a Tensor of size C. Otherwise, it is
                treated as if having all ones. Defaults to None.
            gamma (float, optional): smoothing parameter. Defaults to 2.0.
            reduction (str, optional): Specifies the reduction to apply to the output:
                'none' | 'mean' | 'sum'.
                    'none': no reduction will be applied,
                    'mean': the weighted mean of the output is taken,
                    'sum': the output will be summed.
                Note: size_average and reduce are in the process of being deprecated, and in the
                meantime, specifying either of those two args will override reduction.
                Defaults to "none".
        """
        super().__init__()
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:  # noqa: A002
        """Calculate the Focal Loss."""
        log_prob = func.log_softmax(input, dim=-1)
        prob = torch.exp(log_prob)
        return func.nll_loss(
            ((1 - prob) ** self.gamma) * log_prob,
            target,
            weight=self.weight,
            reduction=self.reduction,
        )
