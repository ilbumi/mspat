from pathlib import Path

import numpy as np
from biocontacts.interfaces.interface import (
    InterfaceSurfacecDescriptors,
    calculate_interface_surface_descriptors,
    calculate_sasa,
    get_interface_mask,
)
from biotite.structure.io import pdb
from padata.transform.base import BaseTransform


def test_get_interface_mask(test_root: Path, structure_preprocessor: BaseTransform) -> None:
    atomstack = pdb.PDBFile.read(test_root / "data/3nzz.ent").get_structure(
        model=1, extra_fields=["charge"], include_bonds=True
    )
    atoms = structure_preprocessor(atomstack)

    result = get_interface_mask(atoms, atoms.chain_id == "A", atoms.chain_id == "B")

    assert len(result) == 10014
    assert result.sum() == 630


def test_calculate_interface_surface_descriptors(test_root: Path, structure_preprocessor: BaseTransform) -> None:
    atomstack = pdb.PDBFile.read(test_root / "data/3nzz.ent").get_structure(
        model=1, extra_fields=["charge"], include_bonds=True
    )
    atoms = structure_preprocessor(atomstack)

    result = calculate_interface_surface_descriptors(atoms, atoms.chain_id == "A", atoms.chain_id == "B")

    assert isinstance(result, InterfaceSurfacecDescriptors)
    assert isinstance(result.delta_total_sasa, float)
    assert isinstance(result.delta_polar_sasa, float)
    assert isinstance(result.delta_apolar_sasa, float)
    assert result.delta_total_sasa < 0
    assert result.delta_polar_sasa < 0
    assert result.delta_apolar_sasa < 0
    assert np.isclose(result.delta_total_sasa, result.delta_polar_sasa + result.delta_apolar_sasa)


def test_calculate_sasa(test_root: Path, structure_preprocessor: BaseTransform) -> None:
    atomstack = pdb.PDBFile.read(test_root / "data/3nzz.ent").get_structure(
        model=1, extra_fields=["charge"], include_bonds=True
    )
    atoms = structure_preprocessor(atomstack)

    total, polar, apolar = calculate_sasa(atoms)

    assert isinstance(total, float)
    assert isinstance(polar, float)
    assert isinstance(apolar, float)
    assert total > 0
    assert polar > 0
    assert apolar > 0
    assert np.isclose(total, polar + apolar)
