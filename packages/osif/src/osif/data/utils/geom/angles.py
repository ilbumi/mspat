"""Utility functions for data processing."""

import numpy as np


def angle(v1: np.ndarray, v2: np.ndarray, acute: bool) -> float:  # noqa: FBT001
    """Calculate angle between two vectors.

    Args:
        v1 (np.ndarray): first vector.
        v2 (np.ndarray): second vector.
        acute (bool): if true return the acute angle, else return the obtuse angle.

    Returns:
        float: angle between to vectors in radians.
    """
    angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    if acute:
        return angle
    return 2 * np.pi - angle
