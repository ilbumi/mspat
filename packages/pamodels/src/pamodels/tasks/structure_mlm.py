import torch
from padata.tensorclasses.structure import TokenTaskProteinStructure
from padata.vocab.residue import ATOM_NAMES_TO_INDEX, RESIDUES_VOCAB

from pamodels.structure.base import BaseAtomTransformer


class AtomTransformerForMLM(torch.nn.Module):
    """Atom Transformer for Masked Language Modeling (MLM) tasks."""

    def __init__(self, atom_transformer: BaseAtomTransformer):
        """Initialize the AtomTransformerForMLM."""
        super().__init__()
        self.atom_transformer = atom_transformer
        self.head = torch.nn.LazyLinear(len(RESIDUES_VOCAB))
        self.loss_fn = torch.nn.CrossEntropyLoss(reduction="mean")

    def forward(
        self,
        data: TokenTaskProteinStructure,
        return_loss: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for the MLM task."""
        feats, _ = self.atom_transformer(
            feats=data.features,
            coors=data.coords,
            adj_mat=data.edges,
            return_coor_changes=False,
        )
        logits = self.head(feats)
        if return_loss:
            loss = self.calculate_loss(data, logits)
            return logits, loss
        return logits

    def calculate_loss(
        self,
        data: TokenTaskProteinStructure,
        logits: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate the loss for the MLM task."""
        # Flatten the target and logits for loss calculation
        target = data.target.view(-1)
        logits = logits.view(-1, logits.size(-1))
        ca_mask = data.features[:, :, 0] == ATOM_NAMES_TO_INDEX["CA"]
        query_mask = ca_mask.view(-1) & (~data.mask.view(-1))

        if not query_mask.any():
            return torch.tensor(0.0, device=logits.device)

        return self.loss_fn(
            logits[query_mask],
            target[query_mask],
        )
