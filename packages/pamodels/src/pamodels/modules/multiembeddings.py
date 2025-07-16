from collections.abc import Sequence

import torch


class MultiEmbedding(torch.nn.Module):
    def __init__(self, num_embeddings: Sequence[int] | int, embedding_dim: int):
        """Initialize the MultiEmbedding module.

        This module creates multiple embeddings, each corresponding to a different
        index in the input tensor. The embeddings are summed together to produce a
        single output tensor.

        Args:
            num_embeddings (Sequence[int] | int): List of sizes for each embedding.
                If a single integer is provided, it is treated as a list with one element.
            embedding_dim (int): Dimension of each embedding.

        """
        super().__init__()
        if isinstance(num_embeddings, int):
            num_embeddings = [num_embeddings]
        self.embeddings = torch.nn.ModuleList(
            [torch.nn.Embedding(vocab_size, embedding_dim) for vocab_size in num_embeddings]
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass for the MultiEmbedding module.

        Args:
            inputs (torch.Tensor): Input tensor containing indices for embeddings.

        Returns:
            torch.Tensor: Concatenated embeddings for the input indices.

        """
        x: torch.Tensor | None = None
        for i, emb in enumerate(self.embeddings):
            if x is None:
                x = emb(inputs[..., i])
            else:
                x += emb(inputs[..., i])
        assert x is not None, "No embeddings were computed."
        return x
