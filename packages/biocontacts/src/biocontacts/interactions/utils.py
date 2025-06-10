import numpy as np
from biotite.structure import infer_elements
from biotite.structure.info import vdw_radius_protor, vdw_radius_single


def deduplicate_interactions(interactors_a: np.ndarray, interactors_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Deduplicate interactions."""
    interactions = np.stack((interactors_a, interactors_b), axis=1)
    interactions = np.sort(interactions, axis=1)
    interactions = np.unique(interactions, axis=0)
    return interactions[:, 0], interactions[:, 1]


def remove_self_interactions(interactors_a: np.ndarray, interactors_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Remove self interactions."""
    mask = interactors_a != interactors_b
    return interactors_a[mask], interactors_b[mask]


def safe_vdw_radius(res_name: str, atom_name: str, element: str | None = None) -> float:
    """Get best available VdW radius."""
    try:
        radius = vdw_radius_protor(res_name, atom_name)
        if radius is not None:
            return radius
    except KeyError:
        pass
    if element is None:
        element = infer_elements([atom_name])[0]
    return vdw_radius_single(element)
