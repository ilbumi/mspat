import numpy as np
from biotite.structure import AtomArray, AtomArrayStack
from biotite.structure.info import vdw_radius_single
from scipy.spatial.distance import cdist

from biocontacts.interactions.interaction import HydrophobicInteraction


def find_hydrophobic_interactions(
    atoms: AtomArray | AtomArrayStack,
    selection1: np.ndarray | None = None,
    selection2: np.ndarray | None = None,
    min_dist: float = 0.3,
    max_dist: float = 0.7,
) -> list[HydrophobicInteraction]:
    """Get VdW interacting residues.

    The function expects a preprocessed protein (use `biostruct.preprocessing.preprocess_protein`).
    The function returns only interactions between selections.
    If selections intersection is not empty, the resulting interaction may be duplicated.
    Only C, S atoms are considered.

    Args:
        atoms (AtomArray | AtomArrayStack): structure. Will use the first structure if AtomArrayStack.
        selection1 (np.ndarray | None, optional): selection 1. None means all atoms. Defaults to None.
        selection2 (np.ndarray | None, optional): selection 2. None means all atoms. Defaults to None.
        min_dist (float, optional): Minimal length of the interaction (between surfaces). Defaults to 0.3.
        max_dist (float, optional): Maximal length of the interaction (between surfaces). Defaults to 0.7.

    Returns:
        list[HydrophobicInteraction]: list of interactions between atoms.

    """
    if isinstance(atoms, AtomArrayStack):
        atoms = atoms[0]
    if selection1 is None:
        selection1 = (atoms.element == "C") | (atoms.element == "S")
    else:
        selection1 = ((atoms.element == "C") | (atoms.element == "S")) & selection1

    if selection2 is None:
        selection2 = (atoms.element == "C") | (atoms.element == "S")
    else:
        selection2 = ((atoms.element == "C") | (atoms.element == "S")) & selection2

    distances = cdist(atoms.coord[selection1], atoms.coord[selection2])
    distances -= np.array([vdw_radius_single(x) for x in atoms.element[selection1]]).reshape(-1, 1)
    distances -= np.array([vdw_radius_single(x) for x in atoms.element[selection2]]).reshape(1, -1)
    row, col = ((distances < max_dist) & (distances > min_dist)).nonzero()

    coord1 = atoms.coord[selection1][row]
    coord2 = atoms.coord[selection2][col]
    atom_names1 = atoms.atom_name[selection1][row]
    atom_names2 = atoms.atom_name[selection2][col]
    chain_ids1 = atoms.chain_id[selection1][row]
    chain_ids2 = atoms.chain_id[selection2][col]
    res_ids1 = atoms.res_id[selection1][row]
    res_ids2 = atoms.res_id[selection2][col]
    res_names1 = atoms.res_name[selection1][row]
    res_names2 = atoms.res_name[selection2][col]
    return [
        HydrophobicInteraction(
            centers=(coord1[i], coord2[i]),
            residue_names=(res_names1[i], res_names2[i]),
            atom_names=(atom_names1[i], atom_names2[i]),
            chain_id_0=str(chain_ids1[i]),
            chain_id_1=str(chain_ids2[i]),
            residue_id_0=int(res_ids1[i]),
            residue_id_1=int(res_ids2[i]),
        )
        for i in range(len(coord1))
    ]
