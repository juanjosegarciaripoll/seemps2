import numpy as np
from seemps.cross import RegularHalfOpenInterval, Mesh, Cross, cross_interpolation
from seemps.state import MPS

from .tools import TestCase


class TestCross(TestCase):
    @staticmethod
    def gaussian_setting(dims):
        a = -1
        b = 1
        n = 5
        func = lambda x: np.exp(-np.sum(x**2))
        intervals = [RegularHalfOpenInterval(a, b, 2**n) for _ in range(dims)]
        mesh = Mesh(intervals)
        mesh_tensor = mesh.to_tensor()
        func_vector = np.apply_along_axis(func, -1, mesh_tensor).flatten()
        mps = MPS.from_vector(func_vector, [2] * (n * dims), normalize=False)
        return func, mesh, mps, func_vector

    # 1D Gaussian
    def test_cross_1d_from_random(self):
        func, mesh, _, func_vector = self.gaussian_setting(1)
        cross = Cross(func, mesh)
        cross, _ = cross_interpolation(cross)
        mps = cross.mps0
        self.assertSimilar(func_vector, mps.to_vector())

    def test_cross_1d_from_mps(self):
        func, mesh, mps0, func_vector = self.gaussian_setting(1)
        mps = Cross(func, mesh, mps0=mps0).run()
        self.assertSimilar(func_vector, mps.to_vector())

    def test_cross_1d_with_measure_norm(self):
        options = {"measure_type": "norm"}
        func, mesh, _, func_vector = self.gaussian_setting(1)
        mps = Cross(func, mesh, options=options).run()
        self.assertSimilar(func_vector, mps.to_vector())

    # 2D Gaussian
    def test_cross_2d_from_random(self):
        func, mesh, _, func_vector = self.gaussian_setting(2)
        mps = Cross(func, mesh).run()
        self.assertSimilar(func_vector, mps.to_vector())

    def test_cross_2d_with_ordering_B(self):
        options = {"ordering": "B"}
        func, mesh, _, func_vector = self.gaussian_setting(2)
        mps = Cross(func, mesh, options=options).run()
        mps_vector = Cross.reorder_tensor(mps.to_vector(), mesh.qubits).flatten()
        self.assertSimilar(func_vector, mps_vector)

    def test_cross_2d_with_structure_tt(self):
        options = {"structure": "tt"}
        func, mesh, _, func_vector = self.gaussian_setting(2)
        mps1 = Cross(func, mesh, options=options).run()
        self.assertSimilar(func_vector, mps1.to_vector())
