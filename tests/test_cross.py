import numpy as np
from seemps.analysis import (
    Mesh,
    RegularClosedInterval,
    RegularHalfOpenInterval,
    ChebyshevZerosInterval,
)
from seemps.cross import CrossStrategy, cross_interpolation, reorder_tensor
from seemps.state import MPS

from .tools import TestCase

"""
Issues (TODO):
    1. The simplification routine simplify() used to truncate the resulting
    MPS changes them a lot and gives incorrect results.
"""


class TestCross(TestCase):
    @staticmethod
    def gaussian_setting(dims, structure="binary"):
        a = -1
        b = 1
        n = 5
        func = lambda x: np.exp(-np.sum(x**2))
        intervals = [RegularHalfOpenInterval(a, b, 2**n) for _ in range(dims)]
        mesh = Mesh(intervals)
        mesh_tensor = mesh.to_tensor()
        func_vector = np.apply_along_axis(func, -1, mesh_tensor).flatten()
        if structure == "binary":
            mps = MPS.from_vector(func_vector, [2] * (n * dims), normalize=False)
        elif structure == "tt":
            mps = MPS.from_vector(func_vector, [2**n] * dims, normalize=False)
        return func, mesh, mps, func_vector

    # 1D Gaussian
    def test_cross_1d_from_random(self):
        func, mesh, _, func_vector = self.gaussian_setting(1)
        mps = cross_interpolation(func, mesh)
        self.assertSimilar(func_vector, mps.to_vector())

    def test_cross_1d_from_mps(self):
        func, mesh, mps0, func_vector = self.gaussian_setting(1)
        mps = cross_interpolation(func, mesh, mps=mps0)
        self.assertSimilar(func_vector, mps.to_vector())

    def test_cross_1d_with_measure_norm(self):
        cross_strategy = CrossStrategy(error_type="norm")
        func, mesh, _, func_vector = self.gaussian_setting(1)
        mps = cross_interpolation(func, mesh, cross_strategy=cross_strategy)
        self.assertSimilar(func_vector, mps.to_vector())

    # FAILS
    # def test_cross_1d_simplified(self):
    #     func, mesh, _, _ = self.gaussian_setting(1)
    #     mps = cross_interpolation(func, mesh)
    #     mps_simplified = simplify(mps)
    #     self.assertSimilar(mps_simplified.to_vector(), mps.to_vector())

    # 2D Gaussian
    def test_cross_2d_from_random(self):
        func, mesh, _, func_vector = self.gaussian_setting(2)
        mps = cross_interpolation(func, mesh)
        self.assertSimilar(func_vector, mps.to_vector())

    def test_cross_2d_with_ordering_B(self):
        cross_strategy = CrossStrategy(mps_ordering="B")
        func, mesh, _, func_vector = self.gaussian_setting(2)
        mps = cross_interpolation(func, mesh, cross_strategy=cross_strategy)
        qubits = [int(np.log2(s)) for s in mesh.shape()[:-1]]
        tensor = reorder_tensor(mps.to_vector(), qubits)
        self.assertSimilar(func_vector, tensor)

    def test_cross_2d_with_structure_tt(self):
        func, mesh, mps0, func_vector = self.gaussian_setting(2, structure="tt")
        mps = cross_interpolation(func, mesh, mps=mps0)
        self.assertSimilar(func_vector, mps.to_vector())


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
