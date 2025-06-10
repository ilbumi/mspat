from collections.abc import Iterable
from pathlib import Path

from biocontacts.interactions.interaction import PiStackingInteraction
from biocontacts.interactions.pi import find_pi_cation_interactions, find_pi_stacking_interactions
from biotite.structure.io import pdb
from padata.transform.base import BaseTransform


def _check_pi_stacking(interactions: Iterable[PiStackingInteraction]) -> None:
    for interaction in interactions:
        assert interaction.centers[0].shape == (3,)
        assert interaction.centers[1].shape == (3,)
        assert interaction.aromatic_atoms_coordinates[0].shape[1] == 3
        assert interaction.aromatic_atoms_coordinates[1].shape[1] == 3
        assert 0 < interaction.plane_angle < 90
        assert 0 < interaction.shift_angle < 90


def test_pi_stacking_full_structure(test_root: Path, structure_preprocessor: BaseTransform) -> None:
    atomstack = pdb.PDBFile.read(test_root / "data/3nzz.ent").get_structure(
        model=1, extra_fields=["charge"], include_bonds=True
    )
    atomstack = structure_preprocessor(atomstack)

    interactions = find_pi_stacking_interactions(atomstack)
    assert len(interactions) == 24
    assert interactions[0].aromatic_atoms_coordinates[0].shape == (6, 3)
    assert interactions[0].aromatic_atoms_coordinates[1].shape == (6, 3)
    assert abs(interactions[0].distance - 5.47) < 0.01
    assert abs(interactions[0].plane_angle - 43.66) < 0.01
    assert abs(interactions[0].shift_angle - 45.64) < 0.01
    _check_pi_stacking(interactions)


def test_pi_stacking_selection(test_root: Path, structure_preprocessor: BaseTransform) -> None:
    atomstack = pdb.PDBFile.read(test_root / "data/3nzz.ent").get_structure(
        model=1, extra_fields=["charge"], include_bonds=True
    )
    atomstack = structure_preprocessor(atomstack)

    selection1 = atomstack.chain_id == "A"
    selection2 = atomstack.chain_id == "B"
    interactions = find_pi_stacking_interactions(atomstack, selection1, selection2)
    assert len(interactions) == 0


def test_pi_cation_full_structure(test_root: Path, structure_preprocessor: BaseTransform) -> None:
    atomstack = pdb.PDBFile.read(test_root / "data/3nzz.ent").get_structure(
        model=1, extra_fields=["charge"], include_bonds=True
    )
    atomstack = structure_preprocessor(atomstack)

    interactions = sorted(find_pi_cation_interactions(atomstack))
    assert len(interactions) == 3
    assert interactions[0].cation_coordinates.shape == (1, 3)
    assert interactions[0].pi_coordinates.shape == (6, 3)
    assert abs(interactions[0].distance - 3.94) < 0.01
    assert abs(interactions[0].angle - 30.52) < 0.01


def test_pi_cation_selection(test_root: Path, structure_preprocessor: BaseTransform) -> None:
    atomstack = pdb.PDBFile.read(test_root / "data/3nzz.ent").get_structure(
        model=1, extra_fields=["charge"], include_bonds=True
    )
    atomstack = structure_preprocessor(atomstack)

    selection1 = atomstack.chain_id == "A"
    selection2 = atomstack.chain_id == "B"
    interactions = find_pi_cation_interactions(atomstack, selection1, selection2)
    assert len(interactions) == 0
