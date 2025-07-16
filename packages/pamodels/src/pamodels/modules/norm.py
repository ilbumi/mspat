import torch


class LayerNorm(torch.nn.Module):
    def __init__(self, dim: int) -> None:
        """Initialize the LayerNorm module."""
        super().__init__()
        self.gamma = torch.nn.Parameter(torch.ones(dim))
        self.beta: torch.Tensor
        self.register_buffer("beta", torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the LayerNorm module."""
        return torch.nn.functional.layer_norm(x, x.shape[-1:], self.gamma, self.beta)
