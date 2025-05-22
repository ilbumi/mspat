import torch
from tensordict import tensorclass


@tensorclass
class ProteinSequence:
    sequence: torch.LongTensor
    mask: torch.BoolTensor
