import numpy as np
from seemps.cross import (
    Mesh,
    RegularClosedInterval,
    RegularHalfOpenInterval,
    ChebyshevZerosInterval,
)

from .tools import TestCase


class TestMesh(TestCase):
    def test_regular_closed_interval(self):
        interval = RegularClosedInterval(0.0, 10, 11)
        assert interval[5] == 5
        vector = interval.to_vector()
        assert np.all(vector == np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]))

    def test_regular_open_interval(self):
        interval = RegularHalfOpenInterval(0.0, 10, 10)
        assert interval[5] == 5
        vector = interval.to_vector()
        assert np.all(vector == np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))

    def test_chebyshev_zeros_interval(self):
        interval = ChebyshevZerosInterval(-1.0, 1.0, 3)
        value = np.sqrt(3) / 2
        assert np.isclose(interval[0], -value)
        vector = interval.to_vector()
        assert np.allclose(vector, np.array([-value, 0, value]))

    def test_rescaled_chebyshev_zeros_interval(self):
        f = lambda x: 2 * x + 2  # Affine transformation
        interval = ChebyshevZerosInterval(f(-1.0), f(1.0), 3)
        value = np.sqrt(3) / 2
        assert np.isclose(interval[0], f(-value))
        vector = interval.to_vector()
        assert np.allclose(vector, np.array([f(-value), f(0), f(value)]))

    def test_regular_closed_mesh(self):
        dimension = 5
        intervals = [RegularClosedInterval(0.0, 10, 11) for _ in range(dimension)]
        mesh = Mesh(intervals)
        assert np.array_equal(mesh[1, 2, 3, 4, 5], np.array([1, 2, 3, 4, 5]))

    def test_regular_half_open_mesh(self):
        dimension = 5
        intervals = [RegularHalfOpenInterval(0.0, 10, 10) for _ in range(dimension)]
        mesh = Mesh(intervals)
        assert np.array_equal(mesh[1, 2, 3, 4, 5], np.array([1, 2, 3, 4, 5]))

    def test_chebyshev_zeros_mesh(self):
        dimension = 3
        f = lambda x: 2 * x + 2
        value = np.sqrt(3) / 2
        intervals = [
            ChebyshevZerosInterval(f(-1.0), f(1.0), 3) for _ in range(dimension)
        ]
        mesh = Mesh(intervals)
        assert np.allclose(mesh[0, 1, 2], np.array([f(-value), f(0), f(value)]))

    def test_mesh_totensor(self):
        dimension = 2
        intervals = [RegularClosedInterval(0.0, 1.0, 2) for _ in range(dimension)]
        mesh = Mesh(intervals)
        tensor = mesh.to_tensor()
        assert np.array_equal(tensor[0, 0], [0, 0])
        assert np.array_equal(tensor[0, 1], [0, 1])
        assert np.array_equal(tensor[1, 0], [1, 0])
        assert np.array_equal(tensor[1, 1], [1, 1])
