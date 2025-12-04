import numpy as np
import unittest
from abc import abstractmethod
from typing import Callable

from seemps.analysis.mesh import RegularInterval, Mesh
from seemps.analysis.lagrange import (
    mps_lagrange_chebyshev_basic,
    mps_lagrange_chebyshev_rr,
    mps_lagrange_chebyshev_lrr,
)

from .tools_analysis import reorder_tensor
from ..tools import TestCase


def gaussian_setup(dim: int) -> tuple[Callable, Mesh]:
    start, stop = -2, 2
    sites = 6
    interval = RegularInterval(start, stop, 2**sites)
    if dim == 1:
        func = lambda x: np.exp(-(x**2))  # noqa: E731
        domain = Mesh([interval])
    elif dim > 1:
        func = lambda tensor: np.exp(-np.sum(tensor**2, axis=0))  # noqa: E731
        domain = Mesh([interval] * dim)
    else:
        raise ValueError("Invalid dimension")
    return func, domain


class LagrangeTests(TestCase):
    @classmethod
    def setUpClass(cls):
        if cls is LagrangeTests:
            raise unittest.SkipTest(f"Skip {cls} tests, it's a base class")
        super().setUpClass()

    @abstractmethod
    def lagrange_method(self, function, *args, **kwdargs):
        raise Exception("Lagrange method not implemented")

    def test_gaussian_1d(self):
        func, domain = gaussian_setup(1)
        mps = self.lagrange_method(func, domain, order=20)
        Z_exact = func(domain.intervals[0].to_vector())
        Z_test = mps.to_vector()
        self.assertSimilar(Z_exact, Z_test)

    def test_gaussian_2d_serial(self):
        func, domain = gaussian_setup(2)
        mps = self.lagrange_method(func, domain, order=20, mps_order="A")
        Z_exact = func(domain.to_tensor(True))
        Z_test = mps.to_vector().reshape(domain.dimensions)
        self.assertSimilar(Z_exact, Z_test)

    def test_gaussian_2d_interleaved(self):
        func, domain = gaussian_setup(2)
        mps = self.lagrange_method(func, domain, order=20, mps_order="B")
        Z_exact = func(domain.to_tensor(True))
        Z_test = mps.to_vector().reshape(domain.dimensions)
        n = int(np.log2(domain.dimensions[0]))
        Z_test = reorder_tensor(Z_test, [n, n])
        self.assertSimilar(Z_exact, Z_test)


class TestLagrangeBasic(LagrangeTests):
    def lagrange_method(self, function, *args, **kwdargs):
        return mps_lagrange_chebyshev_basic(function, *args, **kwdargs)


class TestLagrangeRankRevealing(LagrangeTests):
    def lagrange_method(self, function, *args, **kwdargs):
        return mps_lagrange_chebyshev_rr(function, *args, **kwdargs)


class TestLagrangeLocalRankRevealing(LagrangeTests):
    def lagrange_method(self, function, *args, **kwdargs):
        kwdargs.setdefault("local_order", 20)
        return mps_lagrange_chebyshev_lrr(function, *args, **kwdargs)
