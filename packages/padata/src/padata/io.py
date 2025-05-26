from gzip import GzipFile
from io import TextIOWrapper
from pathlib import Path
from typing import Any

from biotite.structure import Atom, AtomArray, AtomArrayStack
from biotite.structure.io.pdb import PDBFile
from biotite.structure.io.pdbx import CIFFile, get_structure, set_structure


def read_pdb_file(
    pdb_path: str | Path,
    model: Any | None = None,
    altloc: str = "first",
    extra_fields: Any | None = None,
    include_bonds: bool = False,
) -> AtomArray | AtomArrayStack:
    """Read a PDB file and return the structure object."""
    if isinstance(pdb_path, str):
        pdb_path = Path(pdb_path)
    if str(pdb_path).endswith(".gz"):
        with pdb_path.open(mode="rb") as f:
            pdb_handle = TextIOWrapper(GzipFile(fileobj=f, mode="r"))
            structure = PDBFile.read(pdb_handle).get_structure(
                model=model, altloc=altloc, extra_fields=extra_fields, include_bonds=include_bonds
            )
            pdb_handle.close()
            return structure
    else:
        with pdb_path.open("r") as f:
            return PDBFile.read(f).get_structure(
                model=model, altloc=altloc, extra_fields=extra_fields, include_bonds=include_bonds
            )


def write_pdb_file(pdb_path: str | Path, structure: AtomArray | AtomArrayStack) -> None:
    """Write a structure object to a PDB file."""
    if isinstance(pdb_path, str):
        pdb_path = Path(pdb_path)

    out_file = PDBFile()
    out_file.set_structure(structure)

    if str(pdb_path).endswith(".gz"):
        with pdb_path.open(mode="wb") as f:
            pdb_handle = TextIOWrapper(GzipFile(fileobj=f, mode="w"))
            out_file.write(pdb_handle)
            pdb_handle.close()
    else:
        with pdb_path.open("w") as f:
            out_file.write(f)


def read_cif_file(
    pdb_path: str | Path,
    model: Any | None = None,
    altloc: str = "first",
    extra_fields: Any | None = None,
    include_bonds: bool = False,
) -> AtomArrayStack | AtomArray | Atom:
    """Read a PDBx/mmCIF file and return the structure."""
    if isinstance(pdb_path, str):
        pdb_path = Path(pdb_path)
    if str(pdb_path).endswith(".gz"):
        with pdb_path.open(mode="rb") as f:
            pdb_handle = TextIOWrapper(GzipFile(fileobj=f, mode="r"))
            structure = get_structure(
                CIFFile.read(pdb_handle),
                model=model,
                altloc=altloc,
                extra_fields=extra_fields,
                include_bonds=include_bonds,
            )
            pdb_handle.close()
            return structure
    else:
        with pdb_path.open("r") as f:
            return get_structure(
                CIFFile.read(f), model=model, altloc=altloc, extra_fields=extra_fields, include_bonds=include_bonds
            )


def write_cif_file(pdb_path: str | Path, structure: AtomArray | AtomArrayStack) -> None:
    """Write a structure object to a PDBx/mmCIF file."""
    if isinstance(pdb_path, str):
        pdb_path = Path(pdb_path)

    out_file = CIFFile()
    set_structure(out_file, structure)

    if str(pdb_path).endswith(".gz"):
        with pdb_path.open(mode="wb") as f:
            pdb_handle = TextIOWrapper(GzipFile(fileobj=f, mode="w"))
            out_file.write(pdb_handle)
            pdb_handle.close()
    else:
        with pdb_path.open("w") as f:
            out_file.write(f)
