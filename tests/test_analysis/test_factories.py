import numpy as np
from seemps.analysis.factories import (
    mps_equispaced,
    mps_exponential,
    mps_sin,
    mps_cos,
    RegularHalfOpenInterval,
    RegularClosedInterval,
    ChebyshevZerosInterval,
    mps_interval,
    mps_tensor_sum,
    mps_tensor_product,
)
from seemps.analysis.mesh import (
    RegularHalfOpenInterval,
    RegularClosedInterval,
    ChebyshevZerosInterval,
)
from ..tools import TestCase
from .tools_analysis import reorder_tensor


class TestMPSFactories(TestCase):
    def test_mps_equispaced(self):
        self.assertSimilar(
            mps_equispaced(-1, 1, 5).to_vector(),
            np.linspace(-1, 1, 2**5, endpoint=False),
        )

    def test_mps_exponential(self):
        self.assertSimilar(
            mps_exponential(-1, 1, 5, c=1).to_vector(),
            np.exp(np.linspace(-1, 1, 2**5, endpoint=False)),
        )
        self.assertSimilar(
            mps_exponential(-1, 1, 5, c=-1).to_vector(),
            np.exp(-np.linspace(-1, 1, 2**5, endpoint=False)),
        )

    def test_mps_sin(self):
        self.assertSimilar(
            mps_sin(-1, 1, 5).to_vector(),
            np.sin(np.linspace(-1, 1, 2**5, endpoint=False)),
        )

    def test_mps_cos(self):
        self.assertSimilar(
            mps_cos(-1, 1, 5).to_vector(),
            np.cos(np.linspace(-1, 1, 2**5, endpoint=False)),
        )

    def test_mps_interval(self):
        start = -1
        stop = 1
        sites = 5
        mps_half_open = mps_interval(RegularHalfOpenInterval(start, stop, 2**sites))
        mps_closed = mps_interval(RegularClosedInterval(start, stop, 2**sites))
        mps_zeros = mps_interval(ChebyshevZerosInterval(start, stop, 2**sites))
        zeros = lambda d: np.array(
            [np.cos(np.pi * (2 * k - 1) / (2 * d)) for k in range(d, 0, -1)]
        )
        self.assertSimilar(
            mps_half_open, np.linspace(start, stop, 2**sites, endpoint=False)
        )
        self.assertSimilar(
            mps_closed, np.linspace(start, stop, 2**sites, endpoint=True)
        )
        self.assertSimilar(mps_zeros, zeros(2**sites))


class TestMPSOperations(TestCase):
    def test_tensor_product(self):
        sites = 5
        interval = RegularHalfOpenInterval(-1, 2, 2**sites)
        mps_x = mps_interval(interval)
        X, Y = np.meshgrid(interval.to_vector(), interval.to_vector())
        # Order A
        mps_x_times_y_A = mps_tensor_product([mps_x, mps_x], mps_order="A")
        Z_mps_A = mps_x_times_y_A.to_vector().reshape((2**sites, 2**sites))
        self.assertSimilar(Z_mps_A, X * Y)
        # Order B
        mps_x_times_y_B = mps_tensor_product([mps_x, mps_x], mps_order="B")
        Z_mps_B = mps_x_times_y_B.to_vector().reshape((2**sites, 2**sites))
        Z_mps_B = reorder_tensor(Z_mps_B, [sites, sites])
        self.assertSimilar(Z_mps_B, X * Y)

    def test_tensor_sum(self):
        sites = 5
        interval = RegularHalfOpenInterval(-1, 2, 2**sites)
        mps_x = mps_interval(interval)
        X, Y = np.meshgrid(interval.to_vector(), interval.to_vector())
        # Order A
        mps_x_plus_y_A = mps_tensor_sum([mps_x, mps_x], mps_order="A")
        Z_mps_A = mps_x_plus_y_A.to_vector().reshape((2**sites, 2**sites))
        self.assertSimilar(Z_mps_A, X + Y)
        # Order B
        mps_x_plus_y_B = mps_tensor_sum([mps_x, mps_x], mps_order="B")
        Z_mps_B = mps_x_plus_y_B.to_vector().reshape((2**sites, 2**sites))
        Z_mps_B = reorder_tensor(Z_mps_B, [sites, sites])
        self.assertSimilar(Z_mps_B, X + Y)
