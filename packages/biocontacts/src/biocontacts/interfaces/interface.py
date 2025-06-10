from dataclasses import dataclass

import numpy as np
from biotite.structure import AtomArray, sasa

from biocontacts.interactions.base import get_radius_edges


def get_interface_mask(
    atoms: AtomArray, selection1: np.ndarray, selection2: np.ndarray, max_dist: float = 5.0
) -> np.ndarray:
    """Get a boolean mask for interface atoms.

    Args:
        atoms (AtomArray): The atom array.
        selection1 (np.ndarray): The selection of the first interactor.
        selection2 (np.ndarray): The selection of the second interactor.
        max_dist (float, optional): The maximal distance between atoms. Defaults to 5.0.

    Returns:
        np.ndarray: A boolean mask for interface atoms.

    """
    edges, _ = get_radius_edges(atoms.coord[selection1], atoms.coord[selection2], max_dist)
    row = np.unique(edges[0])
    col = np.unique(edges[1])

    mask = np.zeros(atoms.array_length(), dtype=bool)
    mask[selection1.nonzero()[0][row]] = True
    mask[selection2.nonzero()[0][col]] = True
    return mask


@dataclass
class InterfaceSurfacecDescriptors:
    delta_total_sasa: float
    delta_polar_sasa: float
    delta_apolar_sasa: float


def calculate_sasa(
    atoms: AtomArray,
    probe_radius: float = 1.4,
    atom_filter: np.ndarray | None = None,
) -> tuple[float, float, float]:
    """Calculate the total solvent accessible surface area.

    Args:
        atoms (AtomArray): The atom array.
        probe_radius (float, optional): The radius of the solvent probe. Defaults to 1.4.
        atom_filter (np.ndarray | None, optional): A filter for the atoms. Defaults to None.

    Returns:
        tuple[float, float, float]: The total solvent accessible surface area,
            the polar solvent accessible surface area, and the apolar solvent accessible surface area.

    """
    atom_filter = atoms.atom_name != "H" if atom_filter is None else atom_filter & (atoms.atom_name != "H")
    sasa_array = sasa(atoms, probe_radius=probe_radius, atom_filter=atom_filter)
    atom_filter = atom_filter & (sasa_array > 0)
    total_sasa = float(np.sum(sasa_array[atom_filter]))
    polar_sasa = float(
        np.sum(sasa_array[atom_filter & (atoms.atom_name == "O")])
        + np.sum(sasa_array[atom_filter & (atoms.atom_name == "N")])
    )
    apolar_sasa = total_sasa - polar_sasa
    return total_sasa, polar_sasa, apolar_sasa


def calculate_interface_surface_descriptors(
    atoms: AtomArray,
    selection1: np.ndarray,
    selection2: np.ndarray,
    max_dist: float = 5.0,
) -> InterfaceSurfacecDescriptors:
    """Calculate the interface surface area.

    Args:
        atoms (AtomArray): The atom array.
        selection1 (np.ndarray): The selection of the first interactor.
        selection2 (np.ndarray): The selection of the second interactor.
        max_dist (float, optional): The maximal distance between atoms. Defaults to 10.0.

    Returns:
        InterfaceSurfacecDescriptors: The interface surface area descriptors.

    """
    mask = get_interface_mask(atoms, selection1, selection2, max_dist)
    total_sasa_1, polar_sasa_1, apolar_sasa1 = calculate_sasa(atoms[mask & selection1])
    total_sasa_2, polar_sasa_2, apolar_sasa2 = calculate_sasa(atoms[mask & selection2])
    total_sasa_total, polar_sasa_total, apolar_sasa_total = calculate_sasa(atoms[mask])

    return InterfaceSurfacecDescriptors(
        delta_total_sasa=total_sasa_total - (total_sasa_1 + total_sasa_2),
        delta_polar_sasa=polar_sasa_total - (polar_sasa_1 + polar_sasa_2),
        delta_apolar_sasa=apolar_sasa_total - (apolar_sasa1 + apolar_sasa2),
    )
