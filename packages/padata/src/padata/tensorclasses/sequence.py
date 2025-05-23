import torch
from tensordict import tensorclass


@tensorclass
class ProteinSequence:
    sequence: torch.Tensor  # int, BxN
    mask: torch.Tensor  # bool, BxN

    @classmethod
    def collate(
        cls,
        batch: list["ProteinSequence"],
    ) -> "ProteinSequence":
        """Collate a batch of ProteinSequence objects into a single ProteinSequence object."""
        sequences = torch.nn.utils.rnn.pad_sequence([b.sequence for b in batch], batch_first=True)
        masks = torch.nn.utils.rnn.pad_sequence([b.mask for b in batch], batch_first=True)
        return cls(sequence=sequences, mask=masks)  # type: ignore[call-arg]


@tensorclass
class SeqTaskProteinSequence:
    sequence: torch.Tensor  # int, BxN
    mask: torch.Tensor  # bool, BxN
    target: torch.Tensor  # any, BxT

    @classmethod
    def collate(
        cls,
        batch: list["SeqTaskProteinSequence"],
    ) -> "SeqTaskProteinSequence":
        """Collate a batch of SeqTaskProteinSequence objects into a single SeqTaskProteinSequence object."""
        sequences = torch.nn.utils.rnn.pad_sequence([b.sequence for b in batch], batch_first=True)
        masks = torch.nn.utils.rnn.pad_sequence([b.mask for b in batch], batch_first=True)
        targets = torch.cat([b.target for b in batch], dim=0)
        return cls(sequence=sequences, mask=masks, target=targets)  # type: ignore[call-arg]


@tensorclass
class TokenTaskProteinSequence:
    sequence: torch.Tensor  # int, BxN
    mask: torch.Tensor  # bool, BxN
    target: torch.Tensor  # any, BxNxT

    @classmethod
    def collate(
        cls,
        batch: list["TokenTaskProteinSequence"],
    ) -> "TokenTaskProteinSequence":
        """Collate a batch of TokenTaskProteinSequence objects into a single TokenTaskProteinSequence object."""
        sequences = torch.nn.utils.rnn.pad_sequence([b.sequence for b in batch], batch_first=True)
        masks = torch.nn.utils.rnn.pad_sequence([b.mask for b in batch], batch_first=True)
        targets = torch.nn.utils.rnn.pad_sequence([b.target for b in batch], batch_first=True)
        return cls(sequence=sequences, mask=masks, target=targets)  # type: ignore[call-arg]
