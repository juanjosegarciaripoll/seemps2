import numpy as np
from seemps.cross import (
    RegularHalfOpenInterval,
    Mesh,
    cross_interpolation,
    CrossStrategy,
)
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
        mps, _ = cross_interpolation(func, mesh)
        self.assertSimilar(func_vector, mps.to_vector())

    def test_cross_1d_from_mps(self):
        func, mesh, mps0, func_vector = self.gaussian_setting(1)
        mps, _ = cross_interpolation(func, mesh, mps0=mps0)
        self.assertSimilar(func_vector, mps.to_vector())

    # def test_cross_1d_with_measure_norm(self):
    #     cross_strategy = CrossStrategy(measurement_type="norm")
    #     func, mesh, _, func_vector = self.gaussian_setting(1)
    #     mps, _ = cross_interpolation(func, mesh, cross_strategy=cross_strategy)
    #     self.assertSimilar(func_vector, mps.to_vector())

    # 2D Gaussian
    def test_cross_2d_from_random(self):
        func, mesh, _, func_vector = self.gaussian_setting(2)
        mps, _ = cross_interpolation(func, mesh)
        self.assertSimilar(func_vector, mps.to_vector())

    # def test_cross_2d_with_ordering_B(self):
    #     cross_strategy = CrossStrategy(mps_ordering="B")
    #     func, mesh, _, func_vector = self.gaussian_setting(2)
    #     mps, _ = cross_interpolation(func, mesh, cross_strategy=cross_strategy)
    #     self.assertSimilar(func_vector, mps.to_vector())
