"""Transform for randomly applying a transform with a given probability."""

import torch
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform


class RandomApplyTransform(BaseTransform):
    """Apply a transform with a given probability."""

    def __init__(self, transform: BaseTransform, p: float = 0.5):
        """Initialize the transform.

        Args:
            transform (BaseTransform): The transform to apply.
            p (float, optional): The probability of applying the transform. Defaults to 0.5.
        """
        self.transform = transform
        self.p = p

    def forward(self, data: Data) -> Data:
        """Apply the transforms with a given probability."""
        if torch.rand(1) < self.p:
            return self.transform(data)
        return data
