import numpy as np
from seemps.analysis.factories import (
    mps_heaviside,
    mps_sum_of_exponentials,
    mps_exponential,
    mps_sin,
    mps_cos,
    mps_interval,
)
from seemps.analysis.mesh import RegularInterval, ChebyshevInterval, QuantizedInterval
from ..tools import SeeMPSTestCase


class TestMPSFactories(SeeMPSTestCase):
    def test_mps_heaviside(self):
        interval = QuantizedInterval(0, 3, 2, endpoint_right=True)
        # x = [0, 1, 2, 3]
        self.assertSimilar(mps_heaviside(interval, 0).to_vector(), [1, 1, 1, 1])
        self.assertSimilar(mps_heaviside(interval, 1).to_vector(), [0, 1, 1, 1])
        self.assertSimilar(mps_heaviside(interval, 1.5).to_vector(), [0, 0, 1, 1])
        self.assertSimilar(mps_heaviside(interval, 2).to_vector(), [0, 0, 1, 1])
        self.assertSimilar(mps_heaviside(interval, 2.3).to_vector(), [0, 0, 0, 1])
        self.assertSimilar(mps_heaviside(interval, 3).to_vector(), [0, 0, 0, 1])

    def test_mps_exponential(self):
        interval = QuantizedInterval(-0.3, 1.0, 5)
        x = interval.to_vector()
        self.assertSimilar(mps_exponential(interval, k=1).to_vector(), np.exp(x))
        self.assertSimilar(
            mps_exponential(interval, k=0.5).to_vector(), np.exp(0.5 * x)
        )
        self.assertSimilar(
            mps_exponential(interval, k=-0.5j).to_vector(), np.exp(-0.5j * x)
        )

    def test_mps_sum_of_exponentials(self):
        interval = QuantizedInterval(-0.3, 1.0, 5)
        x = interval.to_vector()
        self.assertSimilar(
            mps_exponential(interval, k=0.5).to_vector(), np.exp(0.5 * x)
        )
        self.assertSimilar(
            mps_sum_of_exponentials(interval, k=[0.5]).to_vector(), np.exp(0.5 * x)
        )
        self.assertSimilar(
            mps_sum_of_exponentials(interval, k=[1, 0.5]).to_vector(),
            np.exp(x) + np.exp(0.5 * x),
        )
        self.assertSimilar(
            mps_sum_of_exponentials(interval, k=[1, 0.5], weights=[2, -3j]).to_vector(),
            2 * np.exp(x) - 3j * np.exp(0.5 * x),
        )

    def test_mps_sin(self):
        self.assertSimilar(
            mps_sin((-1, 1, 5)).to_vector(),
            np.sin(np.linspace(-1, 1, 2**5, endpoint=False)),
        )

    def test_mps_cos(self):
        self.assertSimilar(
            mps_cos((-1, 1, 5)).to_vector(),
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
