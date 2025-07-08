"""Utility functions for data processing."""

import math

import numpy as np
from scipy.spatial.transform import Rotation

TETRAHEDRAL_ANGLE = 1.91
COS_TETRAHEDRAL_ANGLE = np.cos(TETRAHEDRAL_ANGLE)
SIN_TETRAHEDRAL_ANGLE = np.sin(TETRAHEDRAL_ANGLE)


def construct_cb_coordinates(n_coords: np.ndarray, ca_coords: np.ndarray, c_coords: np.ndarray) -> np.ndarray:
    """Get approximate CB coordinates from N, CA, C coordinates.

    Args:
        n_coords (np.ndarray): coordinates of N atom.
        ca_coords (np.ndarray): coordinates of CA atom.
        c_coords (np.ndarray): coordinates of C atom.

    Returns:
        np.ndarray: coordinates of CB atom.
    """
    x = COS_TETRAHEDRAL_ANGLE
    y = (COS_TETRAHEDRAL_ANGLE - COS_TETRAHEDRAL_ANGLE**2) / SIN_TETRAHEDRAL_ANGLE
    z = math.sqrt(1 - x**2 - y**2)
    base_coords = np.array([x, y, z])
    rotation = Rotation.align_vectors(
        [n_coords - ca_coords, c_coords - ca_coords],
        np.array([[1, 0, 0], [COS_TETRAHEDRAL_ANGLE, SIN_TETRAHEDRAL_ANGLE, 0]]),
    )[0]
    return 1.54 * rotation.apply(base_coords) + ca_coords
