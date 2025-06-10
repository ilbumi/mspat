import numpy as np


def get_lengths(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Calculate bond lengths between two sets of coordinates.

    Args:
        x (np.ndarray): coordinates of the first set of atoms
        y (np.ndarray): coordinates of the second set of atoms

    Returns:
        np.ndarray: bond lengths

    """
    return np.linalg.norm(x - y, axis=1)


def get_elements_ohe(
    atoms: np.ndarray,
    elements: list[str],
) -> np.ndarray:
    """Get one-hot encoding of elements.

    Args:
        atoms (np.ndarray): array of elements
        elements (list[str]): list of elements

    Returns:
        np.ndarray: one-hot encoding of elements

    """
    return np.array([atoms == element for element in elements]).T


def get_angle_between_vectors(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Calculate angle between two vectors.

    Args:
        a (np.ndarray): first vector
        b (np.ndarray): second vector

    Returns:
        float: angle in degrees

    """
    if a.ndim == 1:
        a = a.reshape(1, -1)
    if b.ndim == 1:
        b = b.reshape(1, -1)
    cos_angle = np.einsum("ij,ij->i", a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))


def get_angles(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Calculate angles between three sets of coordinates.

    Args:
        x (np.ndarray): coordinates of the first set of atoms
        y (np.ndarray): coordinates of the second set of atoms
        z (np.ndarray): coordinates of the third set of atoms

    Returns:
        np.ndarray: angles in degrees

    """
    a = x - y
    b = z - y
    return get_angle_between_vectors(a, b)


def to_acute(angle: np.ndarray | float) -> np.ndarray | float:
    """Convert angle to acute angle.

    Args:
        angle (float | np.ndarray): angle in degrees

    Returns:
        float | np.ndarray: acute angle in degrees

    """
    return angle if angle <= 90.0 else 180.0 - angle  # noqa: PLR2004
