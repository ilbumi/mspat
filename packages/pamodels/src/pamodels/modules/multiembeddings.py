from collections.abc import Sequence

import torch


class MultiEmbedding(torch.nn.Module):
    def __init__(self, num_embeddings: Sequence[int], embedding_dim: int):
        """Initialize the MultiEmbedding module.

        This module creates multiple embeddings, each corresponding to a different
        index in the input tensor. The embeddings are summed together to produce a
        single output tensor.

        Args:
            num_embeddings (Sequence[int]): List of sizes for each embedding.
            embedding_dim (int): Dimension of each embedding.

        """
        super().__init__()
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
        x = None
        for i, emb in enumerate(self.embeddings):
            if x is None:
                x = emb(inputs[..., i])
            else:
                x += emb(inputs[..., i])
        return x
