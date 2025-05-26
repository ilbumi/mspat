from pathlib import Path

from biotite.structure.io.pdb import PDBFile
from padata.transform.bonds import AddResidueBonds


def test_add_bonds_transform(test_root: Path) -> None:
    transform = AddResidueBonds()
    pdb_file = PDBFile.read(test_root / "data" / "3nzz.ent")
    structure = pdb_file.get_structure(model=1, extra_fields=["charge"], include_bonds=True)
    assert hasattr(structure, "bonds")
    structure = transform(structure)
    assert hasattr(structure, "bonds")

    structure = pdb_file.get_structure(model=1, extra_fields=["charge"], include_bonds=False)
    assert hasattr(structure, "bonds")
    structure = transform(structure)
    assert hasattr(structure, "bonds")
