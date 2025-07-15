import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
from biotite.structure import AtomArray, BondList
from padata.io import read_cif_file
from padata.tensorclasses.structure import TokenTaskProteinStructure
from padata.transform.compose import ComposeTransform
from padata.transform.mask import MaskSpans, RemoveMaskedSideChains
from padata.vocab.residue import ATOM_NAMES_TO_INDEX, MASK_TOKEN, RESIDUE_TO_INDEX
from padatasets.tasks.reconstruct.residues import (
    construct_residue_reconstruction_data,
    get_atom_features,
    get_first_array,
    load_cifs,
)


def test_get_first_array_with_atom_array():
    """Test get_first_array with AtomArray input."""
    atom_array = AtomArray(length=5)
    atom_array.coord = np.zeros((5, 3))
    result = get_first_array(atom_array)
    assert result is atom_array


def test_get_first_array_with_invalid_type():
    """Test get_first_array with invalid input type."""
    with pytest.raises(TypeError, match="Expected AtomArray or AtomArrayStack"):
        get_first_array("invalid")


def test_get_atom_features():
    """Test get_atom_features function."""
    atom_array = AtomArray(length=3)
    atom_array.atom_name = np.array(["CA", "CB", "N"])
    atom_array.res_name = np.array(["ALA", "GLY", "VAL"])
    atom_array.chain_id = np.array(["A", "A", "B"])
    atom_array.mask = np.array([True, False, True])  # Mask GLY

    features = get_atom_features(atom_array)

    assert features.shape == (3, 3)
    assert features.dtype == torch.int64
    assert features[0, 2] == 0  # Chain A = 0
    assert features[2, 2] == 1  # Chain B = 1
    assert features[0, 1] == RESIDUE_TO_INDEX["A"]
    assert features[1, 1] == RESIDUE_TO_INDEX[MASK_TOKEN]


def test_get_atom_features_with_unknown_atom():
    """Test get_atom_features with unknown atom name."""
    atom_array = AtomArray(length=1)
    atom_array.atom_name = ["UNKNOWN"]
    atom_array.res_name = ["ALA"]
    atom_array.chain_id = ["A"]
    atom_array.mask = np.array([True])

    features = get_atom_features(atom_array)

    # Should use "X" index for unknown atoms

    assert features[0, 0] == ATOM_NAMES_TO_INDEX["X"]


def test_get_atom_features_with_unknown_residue():
    """Test get_atom_features with unknown residue."""
    atom_array = AtomArray(length=1)
    atom_array.atom_name = ["CA"]
    atom_array.res_name = ["UNK"]
    atom_array.chain_id = ["A"]
    atom_array.mask = np.array([True])

    features = get_atom_features(atom_array)

    # Should use "X" index for unknown residues
    assert features[0, 1] == RESIDUE_TO_INDEX["X"]


def test_construct_residue_reconstruction_data():
    """Test construct_residue_reconstruction_data function."""
    atom_array = AtomArray(length=2)
    atom_array.coord = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    atom_array.atom_name = np.array(["CA", "CB"])
    atom_array.res_name = np.array(["GLY", "ALA"])
    atom_array.chain_id = np.array(["A", "A"])
    atom_array.bonds = BondList(2, np.array([(0, 1)]))
    atom_array.mask = np.array([True, True])

    result = construct_residue_reconstruction_data(atom_array)

    assert isinstance(result, TokenTaskProteinStructure)
    assert result.coords.shape == (2, 3)
    assert result.features.shape == (2, 3)
    assert result.edges.shape == (1, 2, 2, 1)
    assert torch.allclose(result.coords, torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))


def test_load_cifs_with_valid_file(test_root: Path, structure_preprocessor):
    """Test load_cifs with a valid CIF file."""
    # Copy the test CIF file to a temp directory
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        cif_file = test_root / "data" / "3nzz.cif.gz"
        temp_cif = temp_path / "3nzz.cif.gz"
        temp_cif.write_bytes(cif_file.read_bytes())

        results = list(load_cifs(temp_path, structure_preprocessor))

        assert len(results) == 1
        assert isinstance(results[0], TokenTaskProteinStructure)
        assert results[0].coords.shape[0] > 0  # Should have atoms
        assert results[0].features.shape[0] > 0  # Should have features


def test_load_cifs_with_no_files():
    """Test load_cifs with empty directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        results = list(load_cifs(temp_path))
        assert len(results) == 0


@patch("padatasets.tasks.reconstruct.residues.logger")
def test_load_cifs_with_invalid_file(mock_logger):
    """Test load_cifs with invalid CIF file."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        invalid_cif = temp_path / "invalid.cif.gz"
        invalid_cif.write_text("invalid content")

        # Should skip invalid files and return empty list
        results = list(load_cifs(temp_path))
        assert len(results) == 0

    # Should have logged a warning
    mock_logger.opt.assert_called_once()
    mock_logger.opt.return_value.warning.assert_called_once()


def test_load_cifs_without_preprocessor(test_root: Path):
    """Test load_cifs without structure preprocessor."""
    # Copy the test CIF file to a temp directory
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        cif_file = test_root / "data" / "3nzz.cif.gz"
        temp_cif = temp_path / "3nzz.cif.gz"
        temp_cif.write_bytes(cif_file.read_bytes())

        results = list(load_cifs(temp_path, structure_preprocessor=None))

        assert len(results) == 1
        assert isinstance(results[0], TokenTaskProteinStructure)


def test_chain_id_conversion():
    """Test chain ID conversion to 0-based index."""
    atom_array = AtomArray(length=4)
    atom_array.atom_name = ["CA", "CA", "CA", "CA"]
    atom_array.res_name = ["ALA", "ALA", "ALA", "ALA"]
    atom_array.chain_id = ["A", "B", "C", "Z"]
    atom_array.mask = np.array([True, True, True, True])  # All atoms are valid

    features = get_atom_features(atom_array)

    assert features[0, 2] == 0  # A = 0
    assert features[1, 2] == 1  # B = 1
    assert features[2, 2] == 2  # C = 2
    assert features[3, 2] == 25  # Z = 25


def test_construct_residue_reconstruction_data_with_real_structure(test_root: Path):
    """Test construct_residue_reconstruction_data with real structure data."""
    cif_file = test_root / "data" / "3nzz.cif.gz"
    structure = get_first_array(read_cif_file(cif_file, model=None, altloc="first", include_bonds=True))
    transform = ComposeTransform(
        [
            MaskSpans(
                num_spans=(1, 6),
                span_length=(1, 25),
                random_seed=42,
            ),
            RemoveMaskedSideChains(),
        ]
    )
    result = construct_residue_reconstruction_data(transform.transform(structure))

    assert isinstance(result, TokenTaskProteinStructure)
    assert result.coords.shape[0] < len(structure)
    assert result.features.shape == (result.coords.shape[0], 3)
    assert result.edges.shape == (1, result.coords.shape[0], result.coords.shape[0], 1)
    assert torch.all(result.features[:, 0] >= 0)  # Valid atom indices
    assert torch.all(result.features[:, 1] >= 0)  # Valid residue indices
    assert torch.all(result.features[:, 2] >= 0)  # Valid chain indices
