from biotite.structure import AtomArray, AtomArrayStack, connect_via_residue_names

from .base import BaseTransform


class AddResidueBonds(BaseTransform):
    def __init__(self, custom_bond_dict: dict[str, dict[tuple[str, str], int]] | None = None) -> None:
        """Initialize the transform.

        Args:
            custom_bond_dict (dict[str, dict[tuple[str, str], int]] | None, optional):
                A dictionary of dictionaries: The outer dictionary maps residue names
                to inner dictionaries. The inner dictionary maps tuples of two atom
                names to their respective BondType (represented as integer).
                Defaults to None.

        """
        super().__init__()
        self.custom_bond_dict = custom_bond_dict

    def transform(self, atoms: AtomArray | AtomArrayStack) -> AtomArray | AtomArrayStack:
        """Add default bonds for residues from RCSB database."""
        bond_list = connect_via_residue_names(atoms)
        if self.custom_bond_dict is not None:
            custom_bond_list = connect_via_residue_names(
                atoms, custom_bond_dict=self.custom_bond_dict, inter_residue=True
            )
            bond_list = bond_list.merge(custom_bond_list)

        if hasattr(atoms, "bonds") and atoms.bonds is not None:
            atoms.bonds = atoms.bonds.merge(bond_list)
        else:
            atoms.bonds = bond_list

        return atoms
