from pathlib import Path

from biocontacts.interactions.clash import find_clashes
from biotite.structure.io import pdb
from padata.transform.base import BaseTransform


def test_clashes_full_structure(test_root: Path, structure_preprocessor: BaseTransform) -> None:
    atomstack = pdb.PDBFile.read(test_root / "data/3nzz.ent").get_structure(
        model=1, extra_fields=["charge"], include_bonds=True
    )
    atomstack = structure_preprocessor(atomstack)

    interactions = find_clashes(atomstack)
    assert len(interactions) == 1562
    assert interactions[0].centers[0].shape == (3,)
    assert interactions[0].centers[1].shape == (3,)
    assert abs(interactions[0].distance - 2.10) < 0.01
    assert abs(interactions[0].intersection - 1.09) < 0.01


def test_clashes_selection(test_root: Path, structure_preprocessor: BaseTransform) -> None:
    atomstack = pdb.PDBFile.read(test_root / "data/3nzz.ent").get_structure(
        model=1, extra_fields=["charge"], include_bonds=True
    )
    atomstack = structure_preprocessor(atomstack)

    selection1 = atomstack.chain_id == "A"
    selection2 = atomstack.chain_id == "B"
    interactions = find_clashes(atomstack, selection1, selection2)
    assert len(interactions) == 13
    assert interactions[0].centers[0].shape == (3,)
    assert interactions[0].centers[1].shape == (3,)
    assert abs(interactions[0].distance - 3.61) < 0.01
    assert abs(interactions[0].intersection - 0.15) < 0.01
