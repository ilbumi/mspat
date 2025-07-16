from collections.abc import Callable
from typing import Any

import torch
from torch import nn

from pamodels.modules.activation import GEGLU
from pamodels.modules.norm import LayerNorm


class Block(nn.Module):
    def __init__(
        self,
        attn: Callable[..., tuple[torch.Tensor, torch.Tensor]],
        ff: Callable[..., tuple[torch.Tensor, torch.Tensor]],
    ) -> None:
        """Initialize the Block module."""
        super().__init__()
        self.attn = attn
        self.ff = ff

    def forward(
        self,
        inp: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        coor_changes: Any = None,  # noqa: ARG002
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass for the Block module."""
        feats, coors, mask, edges, adj_mat = inp
        feats, coors = self.attn(feats, coors, edges=edges, mask=mask, adj_mat=adj_mat)
        feats, coors = self.ff(feats, coors)
        return (feats, coors, mask, edges, adj_mat)


class CoorsNorm(nn.Module):
    def __init__(self, eps: float = 1e-8, scale_init: float = 1.0) -> None:
        """Initialize the CoorsNorm module."""
        super().__init__()
        self.eps = eps
        scale = torch.zeros(1).fill_(scale_init)
        self.scale = nn.Parameter(scale)

    def forward(self, coors: torch.Tensor) -> torch.Tensor:
        """Forward pass for the CoorsNorm module."""
        norm = coors.norm(dim=-1, keepdim=True)
        normed_coors = coors / norm.clamp(min=self.eps)
        return normed_coors * self.scale


class Residual(nn.Module):
    def __init__(self, fn: Callable[..., tuple[torch.Tensor, torch.Tensor]]) -> None:
        """Initialize the Residual module."""
        super().__init__()
        self.fn = fn

    def forward(self, feats: torch.Tensor, coors: torch.Tensor, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for the Residual module."""
        feats_out, coors_delta = self.fn(feats, coors, **kwargs)
        return feats + feats_out, coors + coors_delta


class FeedForward(nn.Module):
    def __init__(self, *, dim: int, mult: int = 4, dropout: float = 0.0) -> None:
        """Initialize the FeedForward module."""
        super().__init__()
        inner_dim = int(dim * mult * 2 / 3)

        self.net = nn.Sequential(
            LayerNorm(dim),
            nn.Linear(dim, inner_dim * 2, bias=False),
            GEGLU(),
            LayerNorm(inner_dim),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim, bias=False),
        )

    def forward(self, feats: torch.Tensor, coors: Any) -> tuple[torch.Tensor, Any]:  # noqa: ARG002
        """Forward pass for the FeedForward module."""
        return self.net(feats), 0
