from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import torch
from biotite.structure import AtomArray, AtomArrayStack
from loguru import logger
from padata.io import read_cif_file
from padata.tensorclasses.structure import TokenTaskProteinStructure
from padata.transform.base import BaseTransform
from padata.transform.compose import ComposeTransform
from padata.transform.mask import MaskSpans, RemoveMaskedSideChains
from padata.vocab.residue import ATOM_NAMES_TO_INDEX, RESIDUE_3_TO_1, RESIDUE_TO_INDEX


def get_first_array(structure: Any) -> AtomArray:
    """Get the first AtomArray from a structure or raise an error if not found."""
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
        edges=torch.from_numpy(structure.bonds.adjacency_matrix().astype(np.int64)).unsqueeze(0).unsqueeze(-1),  # type: ignore[call-arg]
        mask=torch.from_numpy(structure.mask).unsqueeze(0),  # type: ignore[call-arg]
        target=torch.tensor(
            [RESIDUE_TO_INDEX.get(RESIDUE_3_TO_1.get(x)) or RESIDUE_TO_INDEX["X"] for x in structure.res_name],
            dtype=torch.int64,
        ).reshape(1, -1, 1),
    )


def load_cifs(
    folder: Path,
    structure_preprocessor: BaseTransform | None = None,
    num_spans: tuple[int, int] = (1, 6),
    span_length: tuple[int, int] = (1, 25),
    random_seed: int = 1337,
) -> Iterator[TokenTaskProteinStructure]:
    """Load CIF files from a folder and yield TokenTaskProteinStructure objects."""
    if structure_preprocessor is None:
        structure_preprocessor = ComposeTransform(
            [
                MaskSpans(
                    num_spans=num_spans,
                    span_length=span_length,
                    random_seed=random_seed,
                ),
                RemoveMaskedSideChains(),
            ]
        )
    else:
        structure_preprocessor = ComposeTransform(
            [
                structure_preprocessor,
                MaskSpans(
                    num_spans=num_spans,
                    span_length=span_length,
                    random_seed=random_seed,
                ),
                RemoveMaskedSideChains(),
            ]
        )
    for cif_file in folder.glob("*.cif.gz"):
        try:
            structure = get_first_array(
                read_cif_file(
                    cif_file,
                    model=None,
                    altloc="first",
                    extra_fields=["charge"],
                    include_bonds=True,
                )
            )
            structure = get_first_array(structure_preprocessor.transform(structure))
        except TypeError as err:
            logger.opt(exception=err).warning(f"Failed to extract a structure {cif_file}, skipping.")
        except OSError as err:
            logger.opt(exception=err).warning(f"Failed to read {cif_file}, skipping.")
        else:
            yield construct_residue_reconstruction_data(structure)
