import numpy as np
from seemps.analysis import Mesh, RegularHalfOpenInterval
from seemps.analysis.cross import cross_interpolation, CrossStrategy
from seemps.state import MPS
from seemps.truncate import simplify, SIMPLIFICATION_STRATEGY

from ..tools import TestCase
from .tools_analysis import reorder_tensor


class TestCrossInterpolation(TestCase):
    @staticmethod
    def gaussian_setup(dims, n=5, a=-1, b=1):
        func = lambda vec: np.exp(-(np.sum(vec, axis=-1) ** 2))
        intervals = [RegularHalfOpenInterval(a, b, 2**n) for _ in range(dims)]
        mesh = Mesh(intervals)
        mesh_tensor = mesh.to_tensor()
        func_vector = func(mesh_tensor).reshape(-1)
        mps = MPS.from_vector(func_vector, [2] * (n * dims), normalize=False)
        return func, mesh, mps, func_vector

    # 1D Gaussian
    def test_cross_1d_from_random(self):
        func, mesh, _, func_vector = self.gaussian_setup(1)
        cross_results = cross_interpolation(func, mesh)
        self.assertSimilar(func_vector, cross_results.state.to_vector())

    def test_cross_1d_from_mps(self):
        func, mesh, mps0, func_vector = self.gaussian_setup(1)
        cross_results = cross_interpolation(func, mesh, initial_state=mps0)
        self.assertSimilar(func_vector, cross_results.state.to_vector())

    def test_cross_1d_simplified(self):
        func, mesh, _, _ = self.gaussian_setup(1)
        cross_results = cross_interpolation(func, mesh)
        mps = cross_results.state
        mps_simplified = simplify(
            mps,
            strategy=SIMPLIFICATION_STRATEGY.replace(normalize=False),
        )
        self.assertSimilar(mps_simplified.to_vector(), mps.to_vector())

    # 2D Gaussian
    def test_cross_2d_from_random(self):
        func, mesh, _, func_vector = self.gaussian_setup(2)
        cross_results = cross_interpolation(func, mesh)
        self.assertSimilar(func_vector, cross_results.state.to_vector())

    def test_cross_2d_with_ordering_B(self):
        func, mesh, _, func_vector = self.gaussian_setup(2)
        cross_strategy = CrossStrategy(order="B")
        cross_results = cross_interpolation(func, mesh, cross_strategy=cross_strategy)
        qubits = [int(np.log2(s)) for s in mesh.dimensions]
        tensor = reorder_tensor(cross_results.state.to_vector(), qubits)
        self.assertSimilar(func_vector, tensor)
