from pathlib import Path

from biocontacts.interactions.hydrophobic import find_hydrophobic_interactions
from biotite.structure.io import pdb
from padata.transform.base import BaseTransform


def test_hydrophobic_full_structure(test_root: Path, structure_preprocessor: BaseTransform) -> None:
    atomstack = pdb.PDBFile.read(test_root / "data/3nzz.ent").get_structure(
        model=1, extra_fields=["charge"], include_bonds=True
    )
    atomstack = structure_preprocessor(atomstack)

    interactions = find_hydrophobic_interactions(atomstack)
    assert len(interactions) == 4980
    assert interactions[0].centers[0].shape == (3,)
    assert interactions[0].centers[1].shape == (3,)
    assert abs(interactions[0].distance - 3.85) < 0.01
    assert abs(interactions[0].surface_to_surface_distance - 0.09) < 0.01


def test_hydrophobic_selection(test_root: Path, structure_preprocessor: BaseTransform) -> None:
    atomstack = pdb.PDBFile.read(test_root / "data/3nzz.ent").get_structure(
        model=1, extra_fields=["charge"], include_bonds=True
    )
    atomstack = structure_preprocessor(atomstack)

    selection1 = atomstack.chain_id == "A"
    selection2 = atomstack.chain_id == "B"
    interactions = find_hydrophobic_interactions(atomstack, selection1, selection2)
    assert len(interactions) == 14
    assert interactions[0].centers[0].shape == (3,)
    assert interactions[0].centers[1].shape == (3,)
    assert abs(interactions[0].distance - 3.93) < 0.01
    assert abs(interactions[0].surface_to_surface_distance - 0.44) < 0.01
