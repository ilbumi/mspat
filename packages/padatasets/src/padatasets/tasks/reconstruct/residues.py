from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch
from biotite.structure import AtomArray, AtomArrayStack
from loguru import logger
from padata.io import read_cif_file
from padata.tensorclasses.structure import TokenTaskProteinStructure
from padata.transform.base import BaseTransform
from padata.vocab.residue import ATOM_NAMES_TO_INDEX, RESIDUE_3_TO_1, RESIDUE_TO_INDEX


def _get_first_array(structure: Any) -> AtomArray:
    if isinstance(structure, AtomArray):
        return structure
    if isinstance(structure, AtomArrayStack):
        array = structure[0]
        if isinstance(array, AtomArray):
            return array
        msg = f"Expected AtomArray, got {type(array)}"
        raise TypeError(msg)
    msg = f"Expected AtomArray or AtomArrayStack, got {type(structure)}"
    raise TypeError(msg)


def get_atom_features(structure: AtomArray) -> torch.Tensor:
    """Get atom features from an AtomArray.

    The features include:
    - Atom name index
    - Residue index
    - Chain index (0-based, A=0, B=1, etc.)
    """
    features = torch.zeros((len(structure), 4), dtype=torch.int64)
    for i, atom_name in enumerate(structure.atom_name):
        features[i, 0] = ATOM_NAMES_TO_INDEX.get(atom_name, ATOM_NAMES_TO_INDEX["X"])
        features[i, 1] = RESIDUE_TO_INDEX[RESIDUE_3_TO_1.get(structure.res_name[i], "X")]
        features[i, 2] = ord(structure.chain_id[i][0].upper()) - 65
    return features


def construct_residue_reconstruction_data(structure: AtomArray) -> TokenTaskProteinStructure:
    """Construct a TokenTaskProteinStructure from an AtomArray."""
    return TokenTaskProteinStructure(
        coords=torch.Tensor(structure.coord),  # type: ignore[call-arg]
        features=get_atom_features(structure),  # type: ignore[call-arg]
        edges=torch.zeros((1, len(structure), len(structure), 0), dtype=torch.int64),  # type: ignore[call-arg]
    )


def load_cifs(folder: Path, structure_preprocessor: BaseTransform | None = None) -> Iterator[TokenTaskProteinStructure]:
    """Load CIF files from a folder and yield TokenTaskProteinStructure objects."""
    for cif_file in folder.glob("*.cif.gz"):
        try:
            structure = _get_first_array(
                read_cif_file(
                    cif_file,
                    model=None,
                    altloc="first",
                    extra_fields=None,
                    include_bonds=True,
                )
            )
            if structure_preprocessor is not None:
                structure = _get_first_array(structure_preprocessor.transform(structure))
        except TypeError as err:
            logger.opt(exception=err).warning(f"Failed to extract a structure {cif_file}, skipping.")
        else:
            yield construct_residue_reconstruction_data(structure)
