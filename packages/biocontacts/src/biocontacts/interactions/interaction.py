from functools import cached_property

import numpy as np
from padata.vocab.residue import ATOM_NAMES_TO_INDEX, RESIDUE_1_TO_3, RESIDUE_TO_INDEX

from biocontacts.interactions.descriptors.common import get_angle_between_vectors, get_angles, to_acute
from biocontacts.interactions.utils import safe_vdw_radius

RESIDUE_3_LETTER_TO_INDEX = {RESIDUE_1_TO_3[k]: v for k, v in RESIDUE_TO_INDEX.items() if k in RESIDUE_1_TO_3}


class BaseInteraction:
    def __init__(
        self,
        centers: tuple[np.ndarray, np.ndarray],
        chain_id_0: str | None = None,
        chain_id_1: str | None = None,
        residue_id_0: int | None = None,
        residue_id_1: int | None = None,
        residue_name_0: str | None = None,
        residue_name_1: str | None = None,
    ) -> None:
        """Initialize BaseInteraction."""
        self.centers = centers
        self.chain_id_0: str | None = chain_id_0
        self.chain_id_1: str | None = chain_id_1
        self.residue_id_0: int | None = residue_id_0
        self.residue_id_1: int | None = residue_id_1
        self.residue_name_0: str | None = residue_name_0
        self.residue_name_1: str | None = residue_name_1

    @cached_property
    def distance(self) -> float:
        """Get distance."""
        return float(np.linalg.norm(self.centers[0] - self.centers[1]))

    def to_dict(self) -> dict[str, str | int | float | list[list[float]] | None]:
        """Convert interaction to dictionary."""
        return {
            "centers": [self.centers[0].tolist(), self.centers[1].tolist()],
            "chain_id_0": self.chain_id_0,
            "residue_name_0": self.residue_name_0,
            "residue_id_0": self.residue_id_0,
            "chain_id_1": self.chain_id_1,
            "residue_name_1": self.residue_name_1,
            "residue_id_1": self.residue_id_1,
            "distance": self.distance,
        }

    def __lt__(self, other: "BaseInteraction") -> bool:
        """Compare interactions."""
        return (self.chain_id_0, self.chain_id_1, self.residue_id_0, self.residue_id_1) < (
            other.chain_id_0,
            other.chain_id_1,
            other.residue_id_0,
            other.residue_id_1,
        )

    def __repr__(self) -> str:
        """Get string representation of the interaction."""
        return (
            f"{self.__class__.__name__}("
            f"chain_id_0={self.chain_id_0}, "
            f"residue_name_0={self.residue_name_0}, "
            f"residue_id_0={self.residue_id_0}, "
            f"chain_id_1={self.chain_id_1}, "
            f"residue_name_1={self.residue_name_1}, "
            f"residue_id_1={self.residue_id_1}, "
            f"distance={self.distance:.2f})"
        )


class HydrogenBond(BaseInteraction):
    def __init__(
        self,
        donor_coordinates: np.ndarray,
        hydrogen_coordinates: np.ndarray,
        acceptor_coordinates: np.ndarray,
        donor_atom_name: str,
        acceptor_atom_name: str,
        chain_id_0: str | None = None,
        chain_id_1: str | None = None,
        residue_id_0: int | None = None,
        residue_id_1: int | None = None,
        residue_name_0: str | None = None,
        residue_name_1: str | None = None,
    ) -> None:
        """Initialize HydrogenBond."""
        super().__init__(
            (hydrogen_coordinates, acceptor_coordinates),
            chain_id_0,
            chain_id_1,
            residue_id_0,
            residue_id_1,
            residue_name_0,
            residue_name_1,
        )
        self.donor_atom_name = donor_atom_name
        self.acceptor_atom_name = acceptor_atom_name
        self.donor_coordinates = donor_coordinates

    @cached_property
    def hydrogen_coordinates(self) -> np.ndarray:
        """Get hydrogen coordinates."""
        return self.centers[0]

    @cached_property
    def acceptor_coordinates(self) -> np.ndarray:
        """Get acceptor coordinates."""
        return self.centers[1]

    @cached_property
    def angle(self) -> float:
        """Get angle."""
        return float(
            get_angles(
                self.donor_coordinates,
                self.hydrogen_coordinates,
                self.acceptor_coordinates,
            )[0]
        )

    @cached_property
    def energy(self) -> float:
        """Get energy of the hydrogen bond."""
        angle = self.angle / 180.0 * np.pi
        return -12.14 * np.cos(angle - np.pi) ** 2 * np.exp(-1.75 * (self.distance - 0.34) ** 2)

    @property
    def features(self) -> np.ndarray:
        """Get features vector."""
        donor_ohe = np.zeros(len(ATOM_NAMES_TO_INDEX), dtype=np.float32)
        donor_ohe[ATOM_NAMES_TO_INDEX.get(self.donor_atom_name, ATOM_NAMES_TO_INDEX["X"])] = 1.0
        acceptor_ohe = np.zeros(len(ATOM_NAMES_TO_INDEX), dtype=np.float32)
        acceptor_ohe[ATOM_NAMES_TO_INDEX.get(self.acceptor_atom_name, ATOM_NAMES_TO_INDEX["X"])] = 1.0
        return np.concat((np.array([self.distance, self.angle], dtype=np.float32), donor_ohe, acceptor_ohe))

    def to_dict(self) -> dict[str, str | int | float | list[list[float]] | None]:
        """Convert hydrogen bond to dictionary."""
        return {
            **super().to_dict(),
            "type": "hydrogen_bond",
            "angle": self.angle,
            "donor_atom_name": self.donor_atom_name,
            "acceptor_atom_name": self.acceptor_atom_name,
            "donor_coordinates": self.donor_coordinates.tolist(),
            "hydrogen_coordinates": self.hydrogen_coordinates.tolist(),
            "acceptor_coordinates": self.acceptor_coordinates.tolist(),
            "energy": self.energy,
        }

    def __repr__(self) -> str:
        """Get string representation of the hydrogen bond."""
        return (
            super().__repr__()[:-1] + f", donor_atom_name={self.donor_atom_name}, "
            f"acceptor_atom_name={self.acceptor_atom_name}, "
            f"angle={self.angle:.2f})"
        )


class PiStackingInteraction(BaseInteraction):
    """Pi-stacking interaction."""

    def __init__(
        self,
        aromatic_atoms_coordinates: tuple[np.ndarray, np.ndarray],
        chain_id_0: str | None = None,
        chain_id_1: str | None = None,
        residue_id_0: int | None = None,
        residue_id_1: int | None = None,
        residue_name_0: str | None = None,
        residue_name_1: str | None = None,
    ) -> None:
        """Initialize PiStackingInteraction."""
        centers = np.array(aromatic_atoms_coordinates[0]).mean(0), np.array(aromatic_atoms_coordinates[1]).mean(0)
        super().__init__(centers, chain_id_0, chain_id_1, residue_id_0, residue_id_1, residue_name_0, residue_name_1)
        self.aromatic_atoms_coordinates = aromatic_atoms_coordinates

    @cached_property
    def normal_vectors(self) -> tuple[np.ndarray, np.ndarray]:
        """Get normal vectors."""
        return (
            np.cross(
                self.aromatic_atoms_coordinates[0][1] - self.aromatic_atoms_coordinates[0][0],
                self.aromatic_atoms_coordinates[0][1] - self.aromatic_atoms_coordinates[0][2],
            ),
            np.cross(
                self.aromatic_atoms_coordinates[1][1] - self.aromatic_atoms_coordinates[1][0],
                self.aromatic_atoms_coordinates[1][1] - self.aromatic_atoms_coordinates[1][2],
            ),
        )

    @cached_property
    def plane_angle(self) -> float:
        """Get angle."""
        return float(to_acute(get_angle_between_vectors(*self.normal_vectors)))

    @cached_property
    def shift_angle(self) -> float:
        """Get angle."""
        return float(
            (
                to_acute(
                    get_angle_between_vectors(
                        self.normal_vectors[0],
                        self.centers[0] - self.centers[1],
                    )
                )
                + to_acute(
                    get_angle_between_vectors(
                        self.normal_vectors[1],
                        self.centers[0] - self.centers[1],
                    )
                )
            )
            / 2
        )

    @cached_property
    def energy(self) -> float:
        """Get energy of the pi-stacking interaction."""
        planar_angle = self.plane_angle / 180.0 * np.pi
        shift_angle = self.shift_angle / 180.0 * np.pi
        offset = np.sin(shift_angle) * self.distance
        return (
            -16.04
            * np.cos(2 * planar_angle) ** 2
            * np.exp(-5.27 * (self.distance - 3.5) ** 2)
            * np.exp(-0.81 * (offset - 2.0) ** 2)
        )

    @property
    def features(self) -> np.ndarray:
        """Get features vector."""
        residue_names_ohe = np.zeros(len(RESIDUE_3_LETTER_TO_INDEX), dtype=np.float32)
        residue_names_ohe[RESIDUE_3_LETTER_TO_INDEX.get(self.residue_name_0, RESIDUE_3_LETTER_TO_INDEX["XAA"])] = 1.0  # type: ignore[arg-type]
        residue_names_ohe[RESIDUE_3_LETTER_TO_INDEX.get(self.residue_name_1, RESIDUE_3_LETTER_TO_INDEX["XAA"])] += 1.0  # type: ignore[arg-type]
        return np.concat(
            (
                np.array(
                    [
                        self.distance,
                        self.plane_angle,
                        self.shift_angle,
                    ],
                    dtype=np.float32,
                ),
                residue_names_ohe,
            )
        )

    def to_dict(self) -> dict[str, str | int | float | list[list[float]] | None]:
        """Convert pi-stacking interaction to dictionary."""
        return {
            **super().to_dict(),
            "type": "pi_stacking",
            "energy": self.energy,
            "shift_angle": self.shift_angle,
            "plane_angle": self.plane_angle,
            "aromatic_atoms_coordinates": [coords.tolist() for coords in self.aromatic_atoms_coordinates],
        }

    def __repr__(self) -> str:
        """Get string representation of the pi-stacking interaction."""
        return super().__repr__()[:-1] + f", plane_angle={self.plane_angle:.2f}, shift_angle={self.shift_angle:.2f})"


class HydrophobicInteraction(BaseInteraction):
    """Hydrophobic interaction."""

    def __init__(
        self,
        centers: tuple[np.ndarray, np.ndarray],
        residue_names: tuple[str, str],
        atom_names: tuple[str, str],
        chain_id_0: str | None = None,
        chain_id_1: str | None = None,
        residue_id_0: int | None = None,
        residue_id_1: int | None = None,
    ) -> None:
        """Initialize HydrophobicInteraction."""
        super().__init__(
            centers, chain_id_0, chain_id_1, residue_id_0, residue_id_1, residue_names[0], residue_names[1]
        )
        self.residue_names = residue_names
        self.atom_names = atom_names

    @cached_property
    def surface_to_surface_distance(self) -> float:
        """Get distance."""
        return (
            self.distance
            - safe_vdw_radius(self.residue_names[0], self.atom_names[0])
            - safe_vdw_radius(self.residue_names[1], self.atom_names[1])
        )

    @property
    def features(self) -> np.ndarray:
        """Get features vector."""
        return np.concat(
            (
                np.array(
                    [
                        self.distance,
                        self.surface_to_surface_distance,
                    ],
                    dtype=np.float32,
                ),
            )
        )

    @cached_property
    def energy(self) -> float:
        """Get energy of the hydrophobic interaction."""
        return -0.23 * np.exp(-0.16 * (self.distance - 4.52) ** 2)

    def to_dict(self) -> dict[str, str | int | float | list[list[float]] | None]:
        """Convert hydrophobic interaction to dictionary."""
        return {
            **super().to_dict(),
            "type": "hydrophobic",
            "energy": self.energy,
            "residue_name_0": self.residue_names[0],
            "atom_name_0": self.atom_names[0],
            "residue_name_1": self.residue_names[1],
            "atom_name_1": self.atom_names[1],
            "surface_to_surface_distance": self.surface_to_surface_distance,
        }

    def __repr__(self) -> str:
        """Get string representation of the hydrophobic interaction."""
        return (
            super().__repr__()[:-1] + f", atom_names={[str(x) for x in self.atom_names]}, "
            f"surface_to_surface_distance={self.surface_to_surface_distance:.2f})"
        )


class IonicInteraction(BaseInteraction):
    def __init__(
        self,
        positive_coordinates: np.ndarray,
        negative_coordinates: np.ndarray,
        positive_residue_name: str,
        negative_residue_name: str,
        chain_id_0: str | None = None,
        chain_id_1: str | None = None,
        residue_id_0: int | None = None,
        residue_id_1: int | None = None,
        residue_name_0: str | None = None,
        residue_name_1: str | None = None,
    ) -> None:
        """Initialize IonicInteraction."""
        centers = np.array(positive_coordinates).mean(0), np.array(negative_coordinates).mean(0)
        super().__init__(centers, chain_id_0, chain_id_1, residue_id_0, residue_id_1, residue_name_0, residue_name_1)
        self.positive_coordinates = positive_coordinates
        self.negative_coordinates = negative_coordinates
        self.positive_residue_name = positive_residue_name
        self.negative_residue_name = negative_residue_name

    @property
    def features(self) -> np.ndarray:
        """Get features vector."""
        residue_names_ohe = np.zeros(len(RESIDUE_3_LETTER_TO_INDEX), dtype=np.float32)
        residue_names_ohe[
            RESIDUE_3_LETTER_TO_INDEX.get(self.positive_residue_name, RESIDUE_3_LETTER_TO_INDEX["XAA"])
        ] = 1.0
        residue_names_ohe[
            RESIDUE_3_LETTER_TO_INDEX.get(self.negative_residue_name, RESIDUE_3_LETTER_TO_INDEX["XAA"])
        ] += 1.0
        return np.concat(
            (
                np.array(
                    [
                        self.distance,
                    ],
                    dtype=np.float32,
                ),
                residue_names_ohe,
            )
        )

    @cached_property
    def energy(self) -> float:
        """Get energy of the ionic interaction."""
        return -25.26 / (26.57 * self.distance)  # Coulomb's law, assuming a dielectric constant of 7.0 for proteins

    def to_dict(self) -> dict[str, str | int | float | list[list[float]] | None]:
        """Convert ionic interaction to dictionary."""
        return {
            **super().to_dict(),
            "type": "ionic",
            "energy": self.energy,
            "positive_coordinates": self.positive_coordinates.tolist(),
            "negative_coordinates": self.negative_coordinates.tolist(),
            "positive_residue_name": self.positive_residue_name,
            "negative_residue_name": self.negative_residue_name,
        }


class CationPiInteraction(BaseInteraction):
    def __init__(
        self,
        cation_coordinates: np.ndarray,
        pi_coordinates: np.ndarray,
        chain_id_0: str | None = None,
        chain_id_1: str | None = None,
        residue_id_0: int | None = None,
        residue_id_1: int | None = None,
        residue_name_0: str | None = None,
        residue_name_1: str | None = None,
    ) -> None:
        """Initialize CationPiInteraction."""
        centers = np.array(cation_coordinates).mean(0), np.array(pi_coordinates).mean(0)
        super().__init__(centers, chain_id_0, chain_id_1, residue_id_0, residue_id_1, residue_name_0, residue_name_1)
        self.cation_coordinates = cation_coordinates
        self.pi_coordinates = pi_coordinates

    @cached_property
    def normal_vector(self) -> np.ndarray:
        """Get normal vector for aromatic plane."""
        return np.cross(
            self.pi_coordinates[0] - self.pi_coordinates[1],
            self.pi_coordinates[0] - self.pi_coordinates[2],
        )

    @cached_property
    def angle(self) -> float:
        """Get angle."""
        return float(to_acute(get_angle_between_vectors(self.normal_vector, self.centers[0] - self.centers[1])))

    @property
    def features(self) -> np.ndarray:
        """Get features vector."""
        residue_names_ohe = np.zeros(len(RESIDUE_3_LETTER_TO_INDEX), dtype=np.float32)
        residue_names_ohe[RESIDUE_3_LETTER_TO_INDEX.get(self.residue_name_0, RESIDUE_3_LETTER_TO_INDEX["XAA"])] = 1.0  # type: ignore[arg-type]
        residue_names_ohe[RESIDUE_3_LETTER_TO_INDEX.get(self.residue_name_1, RESIDUE_3_LETTER_TO_INDEX["XAA"])] += 1.0  # type: ignore[arg-type]
        return np.concat(
            (
                np.array(
                    [
                        self.distance,
                        self.angle,
                    ],
                    dtype=np.float32,
                ),
                residue_names_ohe,
            )
        )

    @cached_property
    def energy(self) -> float:
        """Get energy of the cation-pi interaction."""
        return -16.05 * np.exp(-7.56 * (self.distance - 2.54) ** 2)

    def to_dict(self) -> dict[str, str | int | float | list[list[float]] | None]:
        """Convert cation-pi interaction to dictionary."""
        return {
            **super().to_dict(),
            "type": "cation_pi",
            "angle": self.angle,
            "energy": self.energy,
            "cation_coordinates": self.cation_coordinates.tolist(),
            "pi_coordinates": self.pi_coordinates.tolist(),
        }

    def __repr__(self) -> str:
        """Get string representation of the cation-pi interaction."""
        return super().__repr__()[:-1] + f", angle={self.angle:.2f})"


class ClashInteraction(BaseInteraction):
    """Clash interaction."""

    def __init__(
        self,
        centers: tuple[np.ndarray, np.ndarray],
        residue_names: tuple[str, str],
        atom_names: tuple[str, str],
        chain_id_0: str | None = None,
        chain_id_1: str | None = None,
        residue_id_0: int | None = None,
        residue_id_1: int | None = None,
    ) -> None:
        """Initialize ClashInteraction."""
        super().__init__(
            centers, chain_id_0, chain_id_1, residue_id_0, residue_id_1, residue_names[0], residue_names[1]
        )
        self.residue_names = residue_names
        self.atom_names = atom_names

    @cached_property
    def intersection(self) -> float:
        """Get intersection."""
        return (
            safe_vdw_radius(self.residue_names[0], self.atom_names[0])
            + safe_vdw_radius(self.residue_names[1], self.atom_names[1])
            - self.distance
        )

    @property
    def features(self) -> np.ndarray:
        """Get features vector."""
        return np.concat(
            (
                np.array(
                    [
                        self.distance,
                        self.intersection,
                    ],
                    dtype=np.float32,
                ),
            )
        )

    @cached_property
    def energy(self) -> float:
        """Get energy of the clash interaction."""
        return 6.51 * np.exp(10.71 * (self.intersection - 1.12))

    def to_dict(self) -> dict[str, str | int | float | list[list[float]] | None]:
        """Convert clash interaction to dictionary."""
        return {
            **super().to_dict(),
            "type": "clash",
            "energy": self.energy,
            "residue_name_0": self.residue_names[0],
            "atom_name_0": self.atom_names[0],
            "residue_name_1": self.residue_names[1],
            "atom_name_1": self.atom_names[1],
            "intersection": self.intersection,
        }

    def __repr__(self) -> str:
        """Get string representation of the clash interaction."""
        return super().__repr__()[:-1] + f", intersection={self.intersection:.2f}, atom_names={self.atom_names})"
