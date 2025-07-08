"""AtomTensors class for protein structure representation."""

import os
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import prody
import torch

from osif.data.utils.protein.virtual_cb import construct_cb_coordinates
from osif.data.utils.protein.vocabs import ATOM_NAMES_TO_INDEX, RESIDUE_3_TO_1, RESIDUE_TO_INDEX, AtomPropertyName


class AtomTensors:
    """Protein structure representation as a pair of tensors."""

    def __init__(self) -> None:
        """Initialize an empty AtomTensors object."""
        self.pos = torch.zeros((0, 3), dtype=torch.float16)
        self.labels = torch.zeros((0, 4), dtype=torch.int32)
        self.vcb_constructed = False

    @classmethod
    def from_atom_group(
        cls,
        prody_structure: prody.AtomGroup,
        construct_vcb: bool = True,
    ) -> "AtomTensors":
        """Convert a PDB file to torch tensors.

        Args:
            pdb_file (str | Path): path to the PDB file.

        """
        result = cls()
        protein_only = prody_structure.select("protein and not element H")
        result.pos = torch.from_numpy(protein_only.getCoords().astype(np.float16))
        result.labels = torch.from_numpy(
            np.stack(
                [
                    protein_only.getChindices(),
                    protein_only.getResindices(),
                    [RESIDUE_TO_INDEX[RESIDUE_3_TO_1.get(x) or "X"] for x in protein_only.getResnames()],
                    [ATOM_NAMES_TO_INDEX.get(x) or ATOM_NAMES_TO_INDEX["X"] for x in protein_only.getNames()],
                ],
            ).astype(np.int32),
        ).T  # MUST BE CONSISTENT WITH AtomPropertyName
        if construct_vcb:
            # Construct the virtual CB atoms for glycines
            vcb_pos, vcb_labels = AtomTensors.get_vcb_tensors(prody_structure)
            result.pos = torch.cat([result.pos, vcb_pos], dim=0)
            result.labels = torch.cat([result.labels, vcb_labels], dim=0)

        return result

    @staticmethod
    def construct_labels(properties: dict[AtomPropertyName, list[int]]) -> torch.Tensor:
        """Construct labels tensor for the atoms.

        Args:
            properties (dict[str, torch.Tensor]): properties for the atoms.
                All atom properties must be present.

        Returns:
            torch.Tensor: labels tensor.

        """
        return torch.tensor(
            [properties[atom_property] for atom_property in AtomPropertyName],
            dtype=torch.int32,
        ).T

    def save(self, prefix: str | Path) -> None:
        """Save protein tensors to disk.

        Args:
            prefix (str | Path): folder with serialized tensors.

        """
        os.makedirs(prefix, exist_ok=True)
        torch.save(
            self.pos,
            os.path.join(prefix, "pos.pt"),
            pickle_protocol=pickle.HIGHEST_PROTOCOL,
        )
        torch.save(
            self.labels,
            os.path.join(prefix, "labels.pt"),
            pickle_protocol=pickle.HIGHEST_PROTOCOL,
        )

    @classmethod
    def load(cls, prefix: str | Path) -> "AtomTensors":
        """Load protein tensors from disk.

        Args:
            prefix (str | Path): folder with serialized tensors.

        """
        result = cls()
        result.pos = torch.load(os.path.join(prefix, "pos.pt"))
        result.labels = torch.load(os.path.join(prefix, "labels.pt"))
        return result

    @staticmethod
    def get_vcb_tensors(
        prody_structure: prody.AtomGroup,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Construct the virtual CB atoms for glycines.

        Args:
            prody_structure (prody.AtomGroup): prody protein structure.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: positions (float16) and labels (int32) tensors.

        """
        pos: list[np.ndarray] = []
        labels: dict[AtomPropertyName, list[int]] = defaultdict(list)
        for res in prody_structure.iterResidues():
            if res.getResname() == "GLY":
                n = res["N"].getCoords()
                ca = res["CA"].getCoords()
                c = res["C"].getCoords()
                cb = construct_cb_coordinates(n, ca, c)
                pos.append(cb)
                labels[AtomPropertyName.chain].append(res.getChindices()[0])
                labels[AtomPropertyName.resindex].append(res.getResindex())
                labels[AtomPropertyName.residue_type].append(RESIDUE_TO_INDEX["G"])
                labels[AtomPropertyName.atom_type].append(ATOM_NAMES_TO_INDEX["VCB"])
        return torch.from_numpy(np.stack(pos).astype(np.float16)), AtomTensors.construct_labels(labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.pos[idx], self.labels[idx]

    def __len__(self) -> int:
        return len(self.pos)
