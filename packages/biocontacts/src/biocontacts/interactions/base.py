from collections.abc import Callable

import numpy as np
from biotite.structure import AtomArray
from padata.vocab.residue import ACIDIC_GROUPS, BASIC_GROUPS
from scipy.spatial import KDTree


class Energy:
    """Energy values for different types of interactions from RING-2.0."""

    IONIC: float = 20.0  # Kj/mol
    HBOND: float = 17.0  # Kj/mol
    PI_CATION: float = 9.6  # Kj/mol
    PI_STACKING: float = 7.0  # Kj/mol
    VDW: float = 6.0  # Kj/mol


def find_atoms(atoms: AtomArray, res_name_to_atom_names: dict[str, list[str]]) -> list[np.ndarray]:
    """Find atoms from specific residues.

    Args:
        atoms (AtomArray): structure of the protein.
        res_name_to_atom_names (dict[str, list[str]]): residues and their atom names to use in calculation.

    Returns:
        list[np.ndarray]: indices of the atoms in the structure.

    """
    indices = []
    for res_name, charged_group_atom_names in res_name_to_atom_names.items():
        mask = atoms.res_name == res_name
        valid_atoms_mask = np.zeros(mask.shape, dtype=mask.dtype)
        for charged_group_atom_name in charged_group_atom_names:
            valid_atoms_mask |= atoms.atom_name == charged_group_atom_name
        mask &= valid_atoms_mask
        residues = set(zip(atoms.res_id[mask], atoms.chain_id[mask], strict=True))
        for res_id, chain_id in residues:
            submask = mask & (atoms.res_id == res_id) & (atoms.chain_id == chain_id)
            indices.append(np.nonzero(submask)[0])
    return indices


def find_acidic_groups(atoms: AtomArray) -> list[np.ndarray]:
    """Find acidic groups.

    Args:
        atoms (AtomArray): structure of the protein

    Returns:
        list[np.ndarray]: indices of the atoms in the structure.

    """
    return find_atoms(atoms, ACIDIC_GROUPS)


def find_basic_groups(atoms: AtomArray) -> list[np.ndarray]:
    """Find basic groups.

    Args:
        atoms (AtomArray): structure of the protein

    Returns:
        list[np.ndarray]: indices of the atoms in the structure.

    """
    return find_atoms(atoms, BASIC_GROUPS)


def get_radius_edges(
    coordinates_a: np.ndarray, coordinates_b: np.ndarray, max_distance: float, min_distance: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """Get indices of pairs of coordinates that are within a certain distance.

    Args:
        coordinates_a (np.ndarray): coordinates of the first set of points
        coordinates_b (np.ndarray): coordinates of the second set of points
        max_distance (float): maximal distance between points
        min_distance (float, optional): minimal distance between points. Defaults to 0.0.

    Returns:
        np.ndarray: (2, N) array of indices of pairs of points
        np.ndarray: (N,) array of distances between points

    """
    if len(coordinates_a) == 0 or len(coordinates_b) == 0:
        return np.zeros((2, 0), dtype=np.int32), np.zeros(0, dtype=np.float32)

    tree_a = KDTree(coordinates_a)
    tree_b = KDTree(coordinates_b)
    results = tree_a.query_ball_tree(tree_b, r=max_distance)

    edges = np.array([(i, j) for i, js in enumerate(results) for j in js]).T
    if len(edges) == 0:
        return np.zeros((2, 0), dtype=np.int32), np.zeros(0, dtype=np.float32)
    distances = np.linalg.norm(coordinates_a[edges[0]] - coordinates_b[edges[1]], axis=1)
    distances_mask = distances < max_distance
    if min_distance > 0.0:
        distances_mask &= distances > min_distance
    return edges[:, distances_mask], distances[distances_mask]


def find_interactions(
    atoms: AtomArray,
    selection1: np.ndarray | None = None,
    selection2: np.ndarray | None = None,
    filter_func_a: Callable[[AtomArray], np.ndarray] = lambda a: a.coords,
    filter_func_b: Callable[[AtomArray], np.ndarray] = lambda a: a.coords,
    max_dist: float = 5.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Get interacting residues.

    The function expects a preprocessed protein (use `biostruct.preprocessing.default_preprocess_protein`).
    The function returns only interactions between selections.
    If selections' intersection is not empty, the resulting interaction may be duplicated.

    Args:
        atoms (AtomArray): structure
        selection1 (np.ndarray | None, optional): selection 1. None means all atoms. Defaults to None.
        selection2 (np.ndarray | None, optional): selection 2. None means all atoms. Defaults to None.
        filter_func_a (callable[[AtomArray], np.ndarray], optional):
            function to produce the first set of interaction centers (e.g. the acidic groups baricenters).
            See `get_acidic_centers` for an example.
            Defaults to lambda a: a.coords.
        filter_func_b (callable[[AtomArray], np.ndarray], optional):
            function to produce the second set of interaction centers (e.g. the basic groups baricenters).
            See `get_basic_centers` for an example.
            Defaults to lambda a: a.coords.
        max_dist (float, optional): Maximal length of the interaction. Defaults to 5.0.

    Returns:
        tuple[np.ndarray, np.ndarray]: coordinates of the interacting groups from both sides and descriptors.

    """
    if (selection1 is None) and (selection2 is None):
        a_coordinates, a_names = filter_func_a(atoms)
        b_coordinates, b_names = filter_func_b(atoms)
        edge_index = get_radius_edges(a_coordinates, b_coordinates, max_dist)[0]
        return a_names[edge_index[0]], b_names[edge_index[1]]

    selection1 = selection1 if selection1 is not None else np.ones(len(atoms), dtype=bool)
    selection2 = selection2 if selection2 is not None else np.ones(len(atoms), dtype=bool)

    interactor_a_coord1 = filter_func_a(atoms[selection1])
    interactor_b_coord1 = filter_func_b(atoms[selection2])
    edge_index_1 = get_radius_edges(interactor_a_coord1, interactor_b_coord1, max_dist)[0]

    interactor_a_coord2 = filter_func_a(atoms[selection2])
    interactor_b_coord2 = filter_func_b(atoms[selection1])
    edge_index_2 = get_radius_edges(interactor_a_coord2, interactor_b_coord2, max_dist)[0]

    return (
        np.concatenate([interactor_a_coord1[edge_index_1[0]], interactor_a_coord2[edge_index_2[0]]]),
        np.concatenate([interactor_b_coord1[edge_index_1[1]], interactor_b_coord2[edge_index_2[1]]]),
    )
