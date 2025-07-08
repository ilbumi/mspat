"""Transforms for masking central residue."""

from collections.abc import Collection

import torch
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform

from osif.data.utils.protein.vocabs import ATOM_NAMES_TO_INDEX, MASK_TOKEN, RESIDUE_TO_INDEX, AtomPropertyName


class MaskResidueSpan(BaseTransform):
    """Transform for masking the central (queried) residue(s) of the protein and the adjacent residues."""

    def __init__(
        self,
        mask_radius: int = 5,
        atoms_to_leave: Collection[str] = ("N", "CA", "C", "O"),
    ):
        """Initialize the transform.

        Args:
            mask_radius (int, optional): mask radius in residues. Defaults to 5.
            atoms_to_leave (Collection[str], optional): names of the atoms to leave in the masked residues. Defaults to ("N", "CA", "C", "O").

        """
        self.mask_radius = mask_radius
        self.atoms_to_leave = atoms_to_leave

    def _get_preserved_atom_types_mask(self, data: Data) -> torch.Tensor:
        mask = torch.zeros(data.labels.shape[0], dtype=torch.bool)
        for atom_name in self.atoms_to_leave:
            mask |= data.labels[:, AtomPropertyName.atom_type] == ATOM_NAMES_TO_INDEX[atom_name]
        return mask

    def _get_same_residue_mask(self, center_pos_idx: int, data: Data) -> torch.Tensor:
        resindex = data.labels[center_pos_idx, AtomPropertyName.resindex]
        chain_id = data.labels[center_pos_idx, AtomPropertyName.chain]
        return (torch.abs(data.labels[:, AtomPropertyName.resindex] - resindex) <= self.mask_radius) & (
            data.labels[:, AtomPropertyName.chain] == chain_id
        )

    def forward(self, data: Data) -> Data:
        """Mask the central (queried) residue(s) of the protein and the adjacent residues."""
        assert data.pos is not None  # noqa: S101
        assert data.labels is not None  # noqa: S101
        assert data.y is not None  # noqa: S101
        assert data.query_mask is not None  # noqa: S101
        assert (  # noqa: S101
            data.edge_index is None
        ), "This transform is not compatible with edge_index. Calculate edges after the masking."
        chosen_residues_mask = torch.zeros(data.pos.shape[0], dtype=torch.bool)
        for atom_idx in data.query_mask.nonzero():
            chosen_residues_mask |= self._get_same_residue_mask(atom_idx, data)

        data.labels[chosen_residues_mask, AtomPropertyName.residue_type] = RESIDUE_TO_INDEX[MASK_TOKEN]
        preserved_atom_types_mask = self._get_preserved_atom_types_mask(data)

        final_mask = ~(chosen_residues_mask & ~preserved_atom_types_mask)
        data.pos = data.pos[final_mask]
        data.labels = data.labels[final_mask]
        data.y = data.y[final_mask]
        data.query_mask = data.query_mask[final_mask]
        if "num_nodes" in data:
            data.num_nodes = len(data.pos)
        return data

    def __repr__(self):
        return f"{self.__class__.__name__}(span_radius={self.mask_radius}, atoms_to_leave={self.atoms_to_leave})"
