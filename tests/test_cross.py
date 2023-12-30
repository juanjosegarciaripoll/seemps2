import numpy as np
from seemps.analysis import (
    Mesh,
    RegularClosedInterval,
    RegularHalfOpenInterval,
    ChebyshevZerosInterval,
)
from seemps.cross import CrossStrategy, cross_interpolation, reorder_tensor
from seemps.cross.cross import Cross
from seemps.state import MPS
from seemps.expectation import scprod
from seemps.truncate import simplify, SIMPLIFICATION_STRATEGY

from .tools import TestCase

"""
Issues (TODO):
    1. The simplification routine simplify() used to truncate the resulting
    MPS changes them a lot and gives incorrect results.
"""


class TestCrossInterpolation(TestCase):
    @staticmethod
    def prepare_cross_instance(dims, strategy=CrossStrategy(), **kwdargs):
        func, mesh, mps, func_vector = TestCrossInterpolation.gaussian_setting(
            dims, **kwdargs
        )
        cross = Cross(func, mesh, mps, strategy)
        return cross

    @staticmethod
    def gaussian_setting(dims, n=5, a=-1, b=1, structure="binary"):
        func = lambda x: np.exp(-np.sum(x**2))
        intervals = [RegularHalfOpenInterval(a, b, 2**n) for _ in range(dims)]
        mesh = Mesh(intervals)
        mesh_tensor = mesh.to_tensor()
        func_vector = np.apply_along_axis(func, -1, mesh_tensor).flatten()
        if structure == "binary":
            mps = MPS.from_vector(func_vector, [2] * (n * dims), normalize=False)
        elif structure == "tt":
            mps = MPS.from_vector(func_vector, [2**n] * dims, normalize=False)
        else:
            raise ValueError(f"Invalid structure {structure}")
        return func, mesh, mps, func_vector

    def test_cross_dimensions(self):
        cross = self.prepare_cross_instance(1, a=-1, b=1, n=5)
        self.assertEqual(cross.sites, 5 * 1)
        cross = self.prepare_cross_instance(2, a=-1, b=1, n=5)
        self.assertEqual(cross.sites, 5 * 2)

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

    def test_cross_1d_simplified(self):
        func, mesh, _, _ = self.gaussian_setting(1)
        mps = cross_interpolation(func, mesh)
        mps_simplified = simplify(
            mps, strategy=SIMPLIFICATION_STRATEGY.replace(normalize=False)
        )
        self.assertSimilar(mps_simplified.to_vector(), mps.to_vector())

    def test_cross_1d_norm2_error(self):
        func, mesh, mps0, func_vector = self.gaussian_setting(1)
        strategy = CrossStrategy(error_type="norm")
        mps = cross_interpolation(func, mesh, mps=mps0, cross_strategy=strategy)
        self.assertSimilar(func_vector, mps.to_vector())

    # 2D Gaussian
    def test_cross_2d_from_random(self):
        func, mesh, _, func_vector = self.gaussian_setting(2)
        mps = cross_interpolation(func, mesh)
        self.assertSimilar(func_vector, mps.to_vector())

    def test_cross_2d_with_ordering_B(self):
        cross_strategy = CrossStrategy(mps_ordering="B")
        func, mesh, _, func_vector = self.gaussian_setting(2)
        mps = cross_interpolation(func, mesh, cross_strategy=cross_strategy)
        qubits = [int(np.log2(s)) for s in mesh.dimensions]
        tensor = reorder_tensor(mps.to_vector(), qubits)
        self.assertSimilar(func_vector, tensor)

    def test_cross_2d_with_structure_tt(self):
        func, mesh, mps0, func_vector = self.gaussian_setting(2, structure="tt")
        mps = cross_interpolation(func, mesh, mps=mps0)
        self.assertSimilar(func_vector, mps.to_vector())
