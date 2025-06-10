from pathlib import Path

from biocontacts.interactions.ionic import find_ionic_interactions
from biotite.structure.io import pdb
from padata.transform.base import BaseTransform


def test_ionic_full_structure(test_root: Path, structure_preprocessor: BaseTransform) -> None:
    atomstack = pdb.PDBFile.read(test_root / "data/3nzz.ent").get_structure(
        model=1, extra_fields=["charge"], include_bonds=True
    )
    atomstack = structure_preprocessor(atomstack)

    interactions = sorted(find_ionic_interactions(atomstack))
    assert len(interactions) == 24
    assert interactions[0].positive_coordinates.shape == (5, 3)
    assert interactions[0].negative_coordinates.shape == (3, 3)
    assert abs(interactions[0].distance - 4.09) < 0.01


def test_ionic_selection(test_root: Path, structure_preprocessor: BaseTransform) -> None:
    atomstack = pdb.PDBFile.read(test_root / "data/3nzz.ent").get_structure(
        model=1, extra_fields=["charge"], include_bonds=True
    )
    atomstack = structure_preprocessor(atomstack)

    selection1 = atomstack.chain_id == "A"
    selection2 = atomstack.chain_id == "B"
    interactions = find_ionic_interactions(atomstack, selection1, selection2)
    assert len(interactions) == 0
