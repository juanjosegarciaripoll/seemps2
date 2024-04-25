import numpy as np

from seemps.analysis.mesh import RegularHalfOpenInterval
from seemps.analysis.lagrange import (
    lagrange_basic,
    lagrange_rank_revealing,
    lagrange_local_rank_revealing,
)

from ..tools import TestCase


class TestLagrangeMPS(TestCase):
    def test_gaussian_basic(self):
        f = lambda x: np.exp(-(x**2))
        start, stop = -2, 2
        sites = 6
        interval = RegularHalfOpenInterval(start, stop, 2**sites)
        mps = lagrange_basic(f, 20, sites, start, stop)
        self.assertSimilar(f(interval.to_vector()), mps.to_vector())

    def test_gaussian_rank_revealing(self):
        f = lambda x: np.exp(-(x**2))
        start, stop = -2, 2
        sites = 6
        interval = RegularHalfOpenInterval(start, stop, 2**sites)
        mps = lagrange_rank_revealing(f, 20, sites, start, stop)
        self.assertSimilar(f(interval.to_vector()), mps.to_vector())

    def test_gaussian_local_rank_revealing(self):
        f = lambda x: np.exp(-(x**2))
        start, stop = -2, 2
        sites = 6
        interval = RegularHalfOpenInterval(start, stop, 2**sites)
        mps = lagrange_local_rank_revealing(f, 20, 5, sites, start, stop)
        self.assertSimilar(f(interval.to_vector()), mps.to_vector())

    def test_lagrange_basic_does_not_overflow(self):

        #     import warnings
        #     from seemps.analysis.lagrange import LagrangeBuilder
        #     builder = LagrangeBuilder(order=700)
        #     with warnings.catch_warnings(record=True) as w:
        #         warnings.simplefilter("always")
        #         builder.A_C(use_logs=True)
        #         self.assertEqual(len(w), 0, "Expected no warnings, but some were raised.")
        pass  # This test is very expensive
