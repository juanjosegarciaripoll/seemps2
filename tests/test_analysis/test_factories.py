import numpy as np
from seemps.analysis.factories import (
    mps_equispaced,
    mps_exponential,
    mps_sin,
    mps_cos,
    mps_interval,
)
from seemps.analysis.mesh import RegularInterval, ChebyshevInterval
from ..tools import TestCase


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
        start, stop, sites = -1, 1, 5
        N = 2**sites
        mps_half_open = mps_interval(RegularInterval(start, stop, N))
        mps_closed = mps_interval(RegularInterval(start, stop, N, endpoint_right=True))

        mps_zeros = mps_interval(ChebyshevInterval(start, stop, N))
        mps_extrema = mps_interval(ChebyshevInterval(start, stop, N, endpoints=True))
        zeros = np.flip(
            [np.cos(np.pi * (2 * k + 1) / (2 * N)) for k in np.arange(0, N)]
        )
        extrema = np.flip([np.cos(np.pi * k / (N - 1)) for k in np.arange(0, N)])

        self.assertSimilar(mps_half_open, np.linspace(start, stop, N, endpoint=False))
        self.assertSimilar(mps_closed, np.linspace(start, stop, N, endpoint=True))
        self.assertSimilar(mps_zeros, zeros)
        self.assertSimilar(mps_extrema, extrema)
