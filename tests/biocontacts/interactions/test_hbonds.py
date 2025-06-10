from pathlib import Path

from biocontacts.interactions.hbond import find_hydrogen_bonds
from biotite.structure.io import pdb
from padata.transform.base import BaseTransform


def _check_hbonds(hydrogen_bonds) -> None:
    for bond in hydrogen_bonds:
        assert bond.donor_coordinates.shape == (3,)
        assert bond.hydrogen_coordinates.shape == (3,)
        assert bond.acceptor_coordinates.shape == (3,)
        assert 0 < bond.angle < 180


def test_hbonds_full_structure(test_root: Path, structure_preprocessor: BaseTransform) -> None:
    atomstack = pdb.PDBFile.read(test_root / "data/3nzz.ent").get_structure(
        model=1, extra_fields=["charge"], include_bonds=True
    )
    atomstack = structure_preprocessor(atomstack)

    hydrogen_bonds = find_hydrogen_bonds(atomstack)
    assert len(hydrogen_bonds) == 306
    assert hydrogen_bonds[0].donor_atom_name == "O"
    assert hydrogen_bonds[0].acceptor_atom_name == "O"
    assert abs(hydrogen_bonds[0].distance - 2.08) < 0.01
    assert abs(hydrogen_bonds[0].angle - 124.06) < 0.01
    _check_hbonds(hydrogen_bonds)


def test_hbonds_selection(test_root: Path, structure_preprocessor: BaseTransform) -> None:
    atomstack = pdb.PDBFile.read(test_root / "data/3nzz.ent").get_structure(
        model=1, extra_fields=["charge"], include_bonds=True
    )
    atomstack = structure_preprocessor(atomstack)

    selection1 = atomstack.chain_id == "A"
    selection2 = atomstack.chain_id == "B"
    hydrogen_bonds = find_hydrogen_bonds(atomstack, selection1, selection2)
    assert len(hydrogen_bonds) == 12
    assert abs(hydrogen_bonds[0].distance - 2.05) < 0.01
    assert abs(hydrogen_bonds[0].angle - 162.40) < 0.01
    _check_hbonds(hydrogen_bonds)
