from pathlib import Path

import numpy as np
import pytest
from biotite.structure import AtomArray, concatenate
from biotite.structure.filter import filter_peptide_backbone
from biotite.structure.io.pdb import PDBFile
from padata.transform.mask import MaskSpans, RemoveMaskedSideChains


def make_atom_array(num_atoms=50, chain_id="A", start_res_id=1):
    atoms = AtomArray(num_atoms)
    atoms.chain_id = np.array([chain_id] * num_atoms)
    atoms.res_id = np.arange(start_res_id, start_res_id + num_atoms)
    return atoms


def test_maskspans_sets_mask_annotation():
    atoms = make_atom_array()
    mask_transform = MaskSpans(num_spans=(1, 1), span_length=(5, 5), random_seed=42)
    atoms = mask_transform.transform(atoms)
    assert "mask" in atoms.get_annotation_categories()
    mask = atoms.mask
    assert mask.dtype == bool
    assert mask.shape == (len(atoms),)


def test_maskspans_masks_correct_number_of_atoms():
    atoms = make_atom_array()
    mask_transform = MaskSpans(num_spans=(1, 1), span_length=(10, 10), random_seed=123)
    atoms = mask_transform.transform(atoms)
    mask = atoms.mask
    # Should mask exactly 10 atoms
    assert np.sum(~mask) == 10


def test_maskspans_multiple_chains():
    atoms_a = make_atom_array(num_atoms=30, chain_id="A", start_res_id=1)
    atoms_b = make_atom_array(num_atoms=20, chain_id="B", start_res_id=100)

    all_atoms = concatenate([atoms_a, atoms_b])
    mask_transform = MaskSpans(num_spans=(2, 2), span_length=(5, 5), random_seed=99)
    all_atoms = mask_transform.transform(all_atoms)
    mask = all_atoms.mask
    # Should mask 5 atoms in each chain, total 10
    assert 5 <= np.sum(~mask) <= 10


def test_maskspans_reproducibility():
    atoms = make_atom_array()
    mask_transform1 = MaskSpans(num_spans=(1, 1), span_length=(5, 5), random_seed=7)
    mask_transform2 = MaskSpans(num_spans=(1, 1), span_length=(5, 5), random_seed=7)
    atoms1 = mask_transform1.transform(atoms.copy())
    atoms2 = mask_transform2.transform(atoms.copy())
    assert np.array_equal(atoms1.mask, atoms2.mask)


def test_removemaskedsidechains_removes_masked_sidechains(test_root: Path):
    pdb_file = PDBFile.read(test_root / "data" / "3nzz.ent")
    atoms = pdb_file.get_structure(model=1, extra_fields=["charge"], include_bonds=True)
    num_atoms_before = len(atoms)
    num_backbone_atoms = len(atoms[filter_peptide_backbone(atoms)])
    mask_transform = MaskSpans(num_spans=(1, 1), span_length=(10, 10), random_seed=7)
    transform = RemoveMaskedSideChains()
    atoms = transform.transform(mask_transform.transform(atoms))
    assert num_atoms_before > len(atoms)
    assert len(atoms[filter_peptide_backbone(atoms)]) == num_backbone_atoms


def test_removemaskedsidechains_raises_without_mask():
    atoms = make_atom_array(num_atoms=5)
    transform = RemoveMaskedSideChains()
    with pytest.raises(ValueError, match="Annotation category 'mask' is not existing"):
        transform.transform(atoms)
