import numpy as np
from biotite.structure import AtomArray, AtomArrayStack, find_aromatic_rings, find_stacking_interactions

from biocontacts.interactions.base import find_basic_groups, get_radius_edges
from biocontacts.interactions.interaction import CationPiInteraction, PiStackingInteraction


def find_pi_stacking_interactions(
    atoms: AtomArray | AtomArrayStack,
    selection1: np.ndarray | None = None,
    selection2: np.ndarray | None = None,
    centroid_cutoff: float = 6.5,
    plane_angle_tol: float = 30,
    shift_angle_tol: float = 30,
) -> list[PiStackingInteraction]:
    """Get pi-stacking interactions.

    The function expects a preprocessed protein (use `biostruct.preprocessing.preprocess_protein`).
    If selections intersection is not empty, the resulting interaction may be duplicated.

    Args:
        atoms (AtomArray | AtomArrayStack): structure. Interactions are calculated only for the first
            structure in the stack.
        selection1 (np.ndarray | None, optional): selection 1.
            The interaction must have at least one atom in the selection. None means all atoms. Defaults to None.
        selection2 (np.ndarray | None, optional): selection 2. None means all atoms. Defaults to None.
        centroid_cutoff (float, optional): Maximal length of the interaction. Defaults to 6.5.
        plane_angle_tol (float, optional): Tolerance for the angle between the planes of the aromatic rings.
            Defaults to 30.
        shift_angle_tol (float, optional): Tolerance for the angle between the normal vector of the aromatic rings.
            Defaults to 30.

    Returns:
        list[PiStackingInteraction]: bonds.

    """
    if isinstance(atoms, AtomArrayStack):
        atoms = atoms[0]
    interactions = find_stacking_interactions(
        atoms,
        centroid_cutoff=centroid_cutoff,
        plane_angle_tol=plane_angle_tol,
        shift_angle_tol=shift_angle_tol,
    )
    result = []
    for aromatic_system_1_indices, aromatic_system_2_indices, _ in interactions:
        if set(aromatic_system_1_indices) & set(aromatic_system_2_indices):
            # skip self-interactions
            continue
        if selection1 is None and selection2 is None:
            result.append(
                PiStackingInteraction(
                    aromatic_atoms_coordinates=(
                        atoms.coord[aromatic_system_1_indices],
                        atoms.coord[aromatic_system_2_indices],
                    ),
                    chain_id_0=str(atoms.chain_id[aromatic_system_1_indices[0]]),
                    chain_id_1=str(atoms.chain_id[aromatic_system_2_indices[0]]),
                    residue_id_0=int(atoms.res_id[aromatic_system_1_indices[0]]),
                    residue_id_1=int(atoms.res_id[aromatic_system_2_indices[0]]),
                    residue_name_0=str(atoms.res_name[aromatic_system_1_indices[0]]),
                    residue_name_1=str(atoms.res_name[aromatic_system_2_indices[0]]),
                )
            )
        elif selection1 is not None and selection2 is not None:
            # guarantee that the first interactor is in selection1
            # and the second interactor is in selection2
            if np.any(selection1[aromatic_system_1_indices]) and np.any(selection2[aromatic_system_2_indices]):
                result.append(
                    PiStackingInteraction(
                        aromatic_atoms_coordinates=(
                            atoms.coord[aromatic_system_1_indices],
                            atoms.coord[aromatic_system_2_indices],
                        ),
                        chain_id_0=str(atoms.chain_id[aromatic_system_1_indices[0]]),
                        chain_id_1=str(atoms.chain_id[aromatic_system_2_indices[0]]),
                        residue_id_0=int(atoms.res_id[aromatic_system_1_indices[0]]),
                        residue_id_1=int(atoms.res_id[aromatic_system_2_indices[0]]),
                        residue_name_0=str(atoms.res_name[aromatic_system_1_indices[0]]),
                        residue_name_1=str(atoms.res_name[aromatic_system_2_indices[0]]),
                    )
                )
            elif np.any(selection1[aromatic_system_2_indices]) and np.any(selection2[aromatic_system_1_indices]):
                result.append(
                    PiStackingInteraction(
                        aromatic_atoms_coordinates=(
                            atoms.coord[aromatic_system_2_indices],
                            atoms.coord[aromatic_system_1_indices],
                        ),
                        chain_id_0=str(atoms.chain_id[aromatic_system_2_indices[0]]),
                        chain_id_1=str(atoms.chain_id[aromatic_system_1_indices[0]]),
                        residue_id_0=int(atoms.res_id[aromatic_system_2_indices[0]]),
                        residue_id_1=int(atoms.res_id[aromatic_system_1_indices[0]]),
                        residue_name_0=str(atoms.res_name[aromatic_system_2_indices[0]]),
                        residue_name_1=str(atoms.res_name[aromatic_system_1_indices[0]]),
                    )
                )
        elif (
            selection1 is not None
            and (np.any(selection1[aromatic_system_1_indices]) or np.any(selection1[aromatic_system_2_indices]))
        ) or (
            selection2 is not None
            and (np.any(selection2[aromatic_system_2_indices]) or np.any(selection2[aromatic_system_1_indices]))
        ):
            result.append(
                PiStackingInteraction(
                    aromatic_atoms_coordinates=(
                        atoms.coord[aromatic_system_1_indices],
                        atoms.coord[aromatic_system_2_indices],
                    ),
                    chain_id_0=str(atoms.chain_id[aromatic_system_1_indices[0]]),
                    chain_id_1=str(atoms.chain_id[aromatic_system_2_indices[0]]),
                    residue_id_0=int(atoms.res_id[aromatic_system_1_indices[0]]),
                    residue_id_1=int(atoms.res_id[aromatic_system_2_indices[0]]),
                    residue_name_0=str(atoms.res_name[aromatic_system_1_indices[0]]),
                    residue_name_1=str(atoms.res_name[aromatic_system_2_indices[0]]),
                )
            )

    return result


def find_pi_cation_interactions(
    atoms: AtomArray | AtomArrayStack,
    selection1: np.ndarray | None = None,
    selection2: np.ndarray | None = None,
    max_dist: float = 5.5,
) -> list[CationPiInteraction]:
    """Get pi-cation interactions.

    The function expects a preprocessed protein (use `biostruct.preprocessing.preprocess_protein`).
    If selections intersection is not empty, the resulting interaction may be duplicated.

    Args:
        atoms (AtomArray | AtomArrayStack): structure
        selection1 (np.ndarray | None, optional): selection 1. None means all atoms. Defaults to None.
        selection2 (np.ndarray | None, optional): selection 2. None means all atoms. Defaults to None.
        max_dist (float, optional): Maximal length of the interaction. Defaults to 5.5.

    Returns:
        list[CationPiInteraction]: pi-cation interactions.

    """
    if isinstance(atoms, AtomArrayStack):
        atoms = atoms[0]
    if selection1 is None:
        selection1 = np.ones(len(atoms), dtype=bool)
    if selection2 is None:
        selection2 = np.ones(len(atoms), dtype=bool)

    aromatic_rings_indices_list = find_aromatic_rings(atoms)
    aromatic_rings_indices_list = [
        aromatic_rings_indices
        for aromatic_rings_indices in aromatic_rings_indices_list
        if np.any(selection1[aromatic_rings_indices]) or np.any(selection2[aromatic_rings_indices])
    ]
    aromatic_centers = np.stack(
        [atoms.coord[aromatic_rings_indices].mean(axis=0) for aromatic_rings_indices in aromatic_rings_indices_list]
    )

    basic_indices_list = find_basic_groups(atoms)
    basic_indices_list = [
        basic_indices
        for basic_indices in basic_indices_list
        if np.any(selection1[basic_indices]) or np.any(selection2[basic_indices])
    ]
    basic_centers = np.stack([atoms.coord[basic_indices].mean(axis=0) for basic_indices in basic_indices_list])

    indices = get_radius_edges(aromatic_centers, basic_centers, max_distance=max_dist, min_distance=0.001)[0].T

    interactions = []
    for aromatic_idx, basic_idx in indices:
        aromatic_indices = aromatic_rings_indices_list[aromatic_idx]
        basic_indices = basic_indices_list[basic_idx]
        # only between selections
        if (np.any(selection1[aromatic_indices]) and np.any(selection2[basic_indices])) or (
            np.any(selection1[basic_indices]) and np.any(selection2[aromatic_indices])
        ):
            interactions.append(
                CationPiInteraction(
                    cation_coordinates=atoms.coord[basic_indices],
                    pi_coordinates=atoms.coord[aromatic_indices],
                    chain_id_0=str(atoms.chain_id[basic_indices[0]]),
                    chain_id_1=str(atoms.chain_id[aromatic_indices[0]]),
                    residue_id_0=int(atoms.res_id[basic_indices[0]]),
                    residue_id_1=int(atoms.res_id[aromatic_indices[0]]),
                    residue_name_0=str(atoms.res_name[basic_indices[0]]),
                    residue_name_1=str(atoms.res_name[aromatic_indices[0]]),
                )
            )
    return interactions
