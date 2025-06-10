from collections.abc import Sequence

import numpy as np
from biotite.structure import AtomArray, AtomArrayStack, hbond

from biocontacts.interactions.interaction import HydrogenBond


def find_hydrogen_bonds(
    atoms: AtomArray | AtomArrayStack,
    selection1: np.ndarray | None = None,
    selection2: np.ndarray | None = None,
    cutoff_dist: float = 2.5,
    acceptor_elements: Sequence[str] = ("O",),
    donor_elements: Sequence[str] = ("O", "N", "S"),
) -> list[HydrogenBond]:
    """Get hydrogen bonds.

    The function expects a protonated protein with annotated partial_charge
    (use `biostruct.preprocessing.preprocess_protein`).

    Args:
        atoms (AtomArray | AtomArrayStack): structure. Hydrogen bonds are calculated
            for the first structure in the stack.
        selection1 (np.ndarray | None, optional): selection 1. None means all atoms. Defaults to None.
        selection2 (np.ndarray | None, optional): selection 2. None means all atoms. Defaults to None.
        cutoff_dist (float, optional): Maximal length of the interaction. Defaults to 2.5.
        acceptor_elements (Sequence[str], optional): Acceptors. Defaults to ("O",).
        donor_elements (Sequence[str], optional): Donors. Defaults to ("O", "N", "S").

    Returns:
        list[HydrogenBond]: hydrogen bonds

    """
    if isinstance(atoms, AtomArrayStack):
        atoms = atoms[0]
    triplets = hbond(
        atoms,
        selection1,
        selection2,
        donor_elements=donor_elements,
        acceptor_elements=acceptor_elements,
        cutoff_dist=cutoff_dist,
    )
    return [
        HydrogenBond(
            donor_coordinates=atoms.coord[triplet[0]],
            hydrogen_coordinates=atoms.coord[triplet[1]],
            acceptor_coordinates=atoms.coord[triplet[2]],
            donor_atom_name=str(atoms.atom_name[triplet[0]]),
            acceptor_atom_name=str(atoms.atom_name[triplet[2]]),
            chain_id_0=str(atoms.chain_id[triplet[0]]),
            chain_id_1=str(atoms.chain_id[triplet[2]]),
            residue_id_0=int(atoms.res_id[triplet[0]]),
            residue_id_1=int(atoms.res_id[triplet[2]]),
            residue_name_0=str(atoms.res_name[triplet[0]]),
            residue_name_1=str(atoms.res_name[triplet[2]]),
        )
        for triplet in triplets
    ]
