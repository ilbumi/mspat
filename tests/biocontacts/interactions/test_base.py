import numpy as np
from biocontacts.interactions.base import get_radius_edges
from scipy.spatial.distance import cdist


def test_get_radius_edges_empty_input():
    coords_a = np.array([])
    coords_b = np.array([])
    edges, distances = get_radius_edges(coords_a, coords_b, 5.0)
    assert edges.shape == (2, 0)
    assert distances.size == 0


def test_get_radius_edges_basic():
    coords_a = np.array([[0, 0, 0], [1, 1, 1]])
    coords_b = np.array([[0, 0, 1], [2, 2, 2]])
    edges, distances = get_radius_edges(coords_a, coords_b, 2.0)

    expected_edges = np.array([[0, 1, 1], [0, 0, 1]])
    expected_distances = cdist(coords_a, coords_b)[expected_edges[0], expected_edges[1]]

    assert np.array_equal(edges, expected_edges)
    assert np.allclose(distances, expected_distances)


def test_get_radius_edges_with_min_distance():
    coords_a = np.array([[0, 0, 0], [1, 1, 1]])
    coords_b = np.array([[0, 0, 1], [2, 2, 2]])
    edges, distances = get_radius_edges(coords_a, coords_b, 2.0, min_distance=1.0)

    expected_edges = np.array([[1, 1], [0, 1]])
    expected_distances = cdist(coords_a, coords_b)[expected_edges[0], expected_edges[1]]

    assert np.array_equal(edges, expected_edges)
    assert np.allclose(distances, expected_distances)
    assert np.all(distances > 1.0)
    assert np.all(distances < 2.0)


def test_get_radius_edges_no_valid_pairs():
    coords_a = np.array([[0, 0, 0]])
    coords_b = np.array([[10, 10, 10]])
    edges, distances = get_radius_edges(coords_a, coords_b, 5.0)
    assert edges.shape == (2, 0)
    assert distances.size == 0
