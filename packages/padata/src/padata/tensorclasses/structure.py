import torch
from tensordict import tensorclass


@tensorclass
class ProteinStructure:
    coords: torch.Tensor  # BxNx3
    features: torch.Tensor  # BxNxF
    edges: torch.Tensor  # BxNxNxE
    mask: torch.Tensor  # BxN
    target: torch.Tensor  # B
