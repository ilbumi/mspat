import torch


class GEGLU(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the GEGLU module."""
        x, gates = x.chunk(2, dim=-1)
        return x * torch.nn.functional.gelu(gates)
