import tempfile
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from biotite.structure import AtomArray, AtomArrayStack
from padata.io import read_cif_file, read_pdb_file, write_cif_file, write_pdb_file


def check_structure_are_same(structure1: AtomArray | AtomArrayStack, structure2: AtomArray | AtomArrayStack) -> None:
    """Check if two structures are the same."""
    assert structure1.coord is not None
    assert structure2.coord is not None
    assert np.allclose(structure1.coord, structure2.coord, atol=1e-4)
    assert np.all(structure1.element == structure2.element)
    assert np.all(structure1.chain_id == structure2.chain_id)
    assert np.all(structure1.res_id == structure2.res_id)
    assert np.all(structure1.res_name == structure2.res_name)
    assert np.all(structure1.hetero == structure2.hetero)
    assert np.all(structure1.ins_code == structure2.ins_code)
    assert np.all(structure1.occupancy == structure2.occupancy)
    assert np.all(structure1.b_factor == structure2.b_factor)
    assert np.all(structure1.charge == structure2.charge)


@pytest.mark.parametrize(
    ("compressed_file", "uncompressed_file", "read_function"),
    [
        pytest.param("data/3nzz.ent.gz", "data/3nzz.ent", read_pdb_file, id="pdb"),
        pytest.param("data/3nzz.cif.gz", "data/3nzz.cif", read_cif_file, id="cif"),
    ],
)
def test_read_gzip(test_root: Path, compressed_file: str, uncompressed_file: str, read_function: Callable) -> None:
    gz_path = test_root / compressed_file
    path = test_root / uncompressed_file
    struct_gz_path = read_function(gz_path, model=1, extra_fields=["charge"], include_bonds=True)
    struct_gz_str = read_function(str(gz_path), model=1, extra_fields=["charge"], include_bonds=True)
    struct_raw_path = read_function(path, model=1, extra_fields=["charge"], include_bonds=True)
    assert struct_gz_path == struct_gz_str
    assert struct_raw_path == struct_gz_path


@pytest.mark.parametrize(
    ("compressed_file", "uncompressed_file", "read_function"),
    [
        pytest.param("data/3nzz.ent.gz", "data/3nzz.ent", read_pdb_file, id="pdb"),
        pytest.param("data/3nzz.cif.gz", "data/3nzz.cif", read_cif_file, id="cif"),
    ],
)
def test_read_anypath(test_root: Path, compressed_file: str, uncompressed_file: str, read_function: Callable) -> None:
    gz_path = test_root / compressed_file
    path = test_root / uncompressed_file
    struct_gz_path = read_function(gz_path, model=1, extra_fields=["charge"], include_bonds=True)
    struct_gz_str = read_function(str(gz_path), model=1, extra_fields=["charge"], include_bonds=True)
    struct_raw_path = read_function(str(path), model=1, extra_fields=["charge"], include_bonds=True)
    assert struct_gz_path == struct_gz_str
    assert struct_raw_path == struct_gz_path


@pytest.mark.parametrize(
    ("input_file", "output_file_extension", "read_function", "write_function"),
    [
        pytest.param("data/3nzz.ent.gz", "pdb", read_pdb_file, write_pdb_file, id="raw_pdb"),
        pytest.param("data/3nzz.cif.gz", "cif", read_cif_file, write_cif_file, id="raw_cif"),
        pytest.param("data/3nzz.ent.gz", "pdb.gz", read_pdb_file, write_pdb_file, id="compressed_pdb"),
        pytest.param("data/3nzz.cif.gz", "cif.gz", read_cif_file, write_cif_file, id="compressed_cif"),
    ],
)
def test_read_write(
    test_root: Path, input_file: str, output_file_extension: str, read_function: Callable, write_function: Callable
) -> None:
    reference_struct = read_function(
        test_root / input_file, model=1, extra_fields=["charge", "b_factor", "occupancy"], include_bonds=True
    )

    # local file
    with tempfile.NamedTemporaryFile(suffix=output_file_extension) as f:
        path = Path(f.name)
        write_function(path, reference_struct)
        struct = read_function(path, model=1, extra_fields=["charge", "b_factor", "occupancy"], include_bonds=True)
        check_structure_are_same(reference_struct, struct)
