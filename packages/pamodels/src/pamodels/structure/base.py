from abc import ABC

import torch


class BaseAtomTransformer(torch.nn.Module, ABC):
    """Base class for atom transformers.

    This class provides a common interface for atom transformers, which are used to
    transform atom features in protein structures.
    """

    def __init__(self):
        """Initialize the BaseAtomTransformer."""
        super().__init__()

    def forward(
        self,
        feats: torch.Tensor,
        coors: torch.Tensor,
        edges: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        adj_mat: torch.Tensor | None = None,
        return_coor_changes: bool = False,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        """Forward pass of the transformer.

        Args:
            feats (torch.Tensor): Input atom features.
            coors (torch.Tensor): Input atom coordinates.
            edges (torch.Tensor | None): Optional edges tensor.
            mask (torch.Tensor | None): Optional mask tensor.
            adj_mat (torch.Tensor | None): Optional adjacency matrix.
            return_coor_changes (bool): Whether to return coordinate changes.
            **kwargs: Additional keyword arguments.

        Returns:
            torch.Tensor: Transformed tensor with updated atom features.

        """
        msg = "Subclasses must implement this method."
        raise NotImplementedError(msg)
