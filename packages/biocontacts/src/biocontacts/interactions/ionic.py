import numpy as np
from biotite.structure import AtomArray, AtomArrayStack

from biocontacts.interactions.base import find_acidic_groups, find_basic_groups, get_radius_edges
from biocontacts.interactions.interaction import IonicInteraction


def find_ionic_interactions(
    atoms: AtomArray,
    selection1: np.ndarray | None = None,
    selection2: np.ndarray | None = None,
    max_dist: float = 5.0,
) -> list[IonicInteraction]:
    """Get ionic interacting residues.

    The function expects a preprocessed protein (use `biostruct.preprocessing.preprocess_protein`).
    The function returns only interactions between selections.
    If selections intersection is not empty, the resulting interaction may be duplicated.

    Args:
        atoms (AtomArray): structure
        selection1 (np.ndarray | None, optional): selection 1. None means all atoms. Defaults to None.
        selection2 (np.ndarray | None, optional): selection 2. None means all atoms. Defaults to None.
        max_dist (float, optional): Maximal length of the interaction. Defaults to 5.0.

    Returns:
        list[IonicInteraction]: coordinates of the interacting centers.

    """
    if isinstance(atoms, AtomArrayStack):
        atoms = atoms[0]
    if selection1 is None:
        selection1 = np.ones(len(atoms), dtype=bool)
    if selection2 is None:
        selection2 = np.ones(len(atoms), dtype=bool)

    acidic_indices_list = find_acidic_groups(atoms)
    acidic_indices_list = [
        acidic_indices
        for acidic_indices in acidic_indices_list
        if np.any(selection1[acidic_indices]) or np.any(selection2[acidic_indices])
    ]
    acidic_centers = np.stack([atoms.coord[acidic_indices].mean(axis=0) for acidic_indices in acidic_indices_list])

    basic_indices_list = find_basic_groups(atoms)
    basic_indices_list = [
        basic_indices
        for basic_indices in basic_indices_list
        if np.any(selection1[basic_indices]) or np.any(selection2[basic_indices])
    ]
    basic_centers = np.stack([atoms.coord[basic_indices].mean(axis=0) for basic_indices in basic_indices_list])

    indices = get_radius_edges(acidic_centers, basic_centers, max_distance=max_dist)[0].T
    interactions = []
    for acidic_idx, basic_idx in indices:
        acidic_indices = acidic_indices_list[acidic_idx]
        basic_indices = basic_indices_list[basic_idx]
        # only between selections
        if (np.any(selection1[acidic_indices]) and np.any(selection2[basic_indices])) or (
            np.any(selection1[basic_indices]) and np.any(selection2[acidic_indices])
        ):
            interactions.append(
                IonicInteraction(
                    positive_coordinates=atoms.coord[basic_indices],
                    negative_coordinates=atoms.coord[acidic_indices],
                    positive_residue_name=str(atoms.res_name[basic_indices[0]]),
                    negative_residue_name=str(atoms.res_name[acidic_indices[0]]),
                    chain_id_0=str(atoms.chain_id[basic_indices[0]]),
                    chain_id_1=str(atoms.chain_id[acidic_indices[0]]),
                    residue_id_0=int(atoms.res_id[basic_indices[0]]),
                    residue_id_1=int(atoms.res_id[acidic_indices[0]]),
                    residue_name_0=str(atoms.res_name[basic_indices[0]]),
                    residue_name_1=str(atoms.res_name[acidic_indices[0]]),
                )
            )
    return interactions
