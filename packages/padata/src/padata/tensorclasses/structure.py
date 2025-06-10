import torch
from tensordict import tensorclass

from padata.utils.pad import pad_matrix


@tensorclass
class ProteinStructure:
    coords: torch.Tensor  # float, BxNx3
    features: torch.Tensor  # any, BxNxF
    edges: torch.Tensor  # any, BxNxNxE
    mask: torch.Tensor  # bool, BxN

    @classmethod
    def collate(
        cls,
        batch: list["ProteinStructure"],
    ) -> "ProteinStructure":
        """Collate a batch of ProteinStructure objects into a single ProteinStructure object."""
        coords = torch.nn.utils.rnn.pad_sequence([b.coords for b in batch], batch_first=True)
        features = torch.nn.utils.rnn.pad_sequence([b.features for b in batch], batch_first=True)
        masks = torch.nn.utils.rnn.pad_sequence([b.mask for b in batch], batch_first=True)
        edges = pad_matrix([b.edges for b in batch], padding_value=0, padding_side="right")
        return cls(coords=coords, features=features, edges=edges, mask=masks)  # type: ignore[call-arg]


@tensorclass
class SeqTaskProteinStructure:
    coords: torch.Tensor  # float, BxNx3
    features: torch.Tensor  # any, BxNxF
    edges: torch.Tensor  # any, BxNxNxE
    mask: torch.Tensor  # bool, BxN
    target: torch.Tensor  # any, BxT

    @classmethod
    def collate(
        cls,
        batch: list["SeqTaskProteinStructure"],
    ) -> "SeqTaskProteinStructure":
        """Collate a batch of SeqTaskProteinStructure objects into a single SeqTaskProteinStructure object."""
        coords = torch.nn.utils.rnn.pad_sequence([b.coords for b in batch], batch_first=True)
        features = torch.nn.utils.rnn.pad_sequence([b.features for b in batch], batch_first=True)
        masks = torch.nn.utils.rnn.pad_sequence([b.mask for b in batch], batch_first=True)
        edges = pad_matrix([b.edges for b in batch], padding_value=0, padding_side="right")
        targets = torch.cat([b.target for b in batch], dim=0)
        return cls(coords=coords, features=features, edges=edges, mask=masks, target=targets)  # type: ignore[call-arg]


@tensorclass
class TokenTaskProteinStructure:
    coords: torch.Tensor  # float, BxNx3
    features: torch.Tensor  # any, BxNxF
    edges: torch.Tensor  # any, BxNxNxE
    mask: torch.Tensor  # bool, BxN
    target: torch.Tensor  # any, BxNxT

    @classmethod
    def collate(
        cls,
        batch: list["TokenTaskProteinStructure"],
    ) -> "TokenTaskProteinStructure":
        """Collate a batch of TokenTaskProteinStructure objects into a single TokenTaskProteinStructure object."""
        coords = torch.nn.utils.rnn.pad_sequence([b.coords for b in batch], batch_first=True)
        features = torch.nn.utils.rnn.pad_sequence([b.features for b in batch], batch_first=True)
        masks = torch.nn.utils.rnn.pad_sequence([b.mask for b in batch], batch_first=True)
        edges = pad_matrix([b.edges for b in batch], padding_value=0, padding_side="right")
        target = torch.nn.utils.rnn.pad_sequence([b.target for b in batch], batch_first=True)
        return cls(coords=coords, features=features, edges=edges, mask=masks, target=target)  # type: ignore[call-arg]
