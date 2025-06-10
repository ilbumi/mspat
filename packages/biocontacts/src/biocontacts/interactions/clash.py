import numpy as np
from biotite.structure import AtomArray, AtomArrayStack
from biotite.structure.info import vdw_radius_single

from biocontacts.interactions.base import get_radius_edges
from biocontacts.interactions.interaction import ClashInteraction
from biocontacts.interactions.utils import safe_vdw_radius


def find_clashes(
    atoms: AtomArray | AtomArrayStack,
    selection1: np.ndarray | None = None,
    selection2: np.ndarray | None = None,
    tolerance: float = 0.15,
) -> list[ClashInteraction]:
    """Get clashes.

    The function expects a preprocessed protein (use `biostruct.preprocessing.preprocess_protein`).
    The function returns only interactions between selections.
    If selections intersection is not empty, the resulting interaction may be duplicated.
    Only C, S atoms are considered.

    Args:
        atoms (AtomArray | AtomArrayStack): structure. Will use the first structure if AtomArrayStack.
        selection1 (np.ndarray | None, optional): selection 1. None means all atoms. Defaults to None.
        selection2 (np.ndarray | None, optional): selection 2. None means all atoms. Defaults to None.
        tolerance (float, optional): Minimal VdW intersection. Defaults to 0.15.

    Returns:
        list[ClashInteraction]: list of interactions between atoms.

    """
    if isinstance(atoms, AtomArrayStack):
        atoms = atoms[0]
    selection1 = atoms.element != "H" if selection1 is None else (atoms.element != "H") & selection1
    selection2 = atoms.element != "H" if selection2 is None else (atoms.element != "H") & selection2

    # get maximal distance between atoms to clash
    all_elements = {str(x) for x in np.unique(atoms.element[selection1])} | {
        str(x) for x in np.unique(atoms.element[selection2])
    }
    max_distance = 2 * max(vdw_radius_single(x) or 0.0 for x in all_elements)

    edges, distances = get_radius_edges(
        atoms.coord[selection1],
        atoms.coord[selection2],
        max_distance=max_distance,
        min_distance=0.001,  # avoid self-clashes
    )
    sum_of_vdw = np.array(
        [
            safe_vdw_radius(res_name, atom_name, element)
            for res_name, atom_name, element in zip(
                atoms.res_name[selection1][edges[0]],
                atoms.atom_name[selection1][edges[0]],
                atoms.element[selection1][edges[0]],
                strict=False,
            )
        ]
    ) + np.array(
        [
            safe_vdw_radius(res_name, atom_name, element)
            for res_name, atom_name, element in zip(
                atoms.res_name[selection2][edges[1]],
                atoms.atom_name[selection2][edges[1]],
                atoms.element[selection2][edges[1]],
                strict=False,
            )
        ]
    )
    idx = (distances + tolerance < sum_of_vdw).nonzero()

    coord1 = atoms.coord[selection1][edges[0][idx]]
    coord2 = atoms.coord[selection2][edges[1][idx]]
    atom_names1 = atoms.atom_name[selection1][edges[0][idx]]
    atom_names2 = atoms.atom_name[selection2][edges[1][idx]]
    chain_ids1 = atoms.chain_id[selection1][edges[0][idx]]
    chain_ids2 = atoms.chain_id[selection2][edges[1][idx]]
    res_ids1 = atoms.res_id[selection1][edges[0][idx]]
    res_ids2 = atoms.res_id[selection2][edges[1][idx]]
    res_names1 = atoms.res_name[selection1][edges[0][idx]]
    res_names2 = atoms.res_name[selection2][edges[1][idx]]
    return [
        ClashInteraction(
            centers=(coord1[i], coord2[i]),
            residue_names=(str(res_names1[i]), str(res_names2[i])),
            atom_names=(str(atom_names1[i]), str(atom_names2[i])),
            chain_id_0=str(chain_ids1[i]),
            chain_id_1=str(chain_ids2[i]),
            residue_id_0=int(res_ids1[i]),
            residue_id_1=int(res_ids2[i]),
        )
        for i in range(len(coord1))
        if not (
            chain_ids1[i] == chain_ids2[i] and abs(res_ids1[i] - res_ids2[i]) <= 2  # noqa: PLR2004
        )  # skip neighboring residues
    ]
