import numpy as np
from scipy.fft import ifft
from seemps.analysis.integrals import (
    mps_midpoint,
    mps_trapezoidal,
    mps_simpson,
    mps_fifth_order,
    mps_fejer,
)
from ..tools import TestCase


class TestMPSIntegrals(TestCase):
    def test_mps_midpoint(self):
        a, b, n = -1, 1, 3
        h = (b - a) / (2**n - 1)
        vector_quad = h * np.array([1, 1, 1, 1, 1, 1, 1, 1])
        mps_quad = mps_midpoint(a, b, n)
        self.assertSimilar(vector_quad, mps_quad.to_vector())

    def test_mps_trapezoidal(self):
        a, b, n = -1, 1, 3
        h = (b - a) / (2**n - 1)
        vector_quad = (h / 2) * np.array([1, 2, 2, 2, 2, 2, 2, 1])
        mps_quad = mps_trapezoidal(-1, 1, 3)
        self.assertSimilar(vector_quad, mps_quad.to_vector())

    def test_mps_simpson(self):
        a, b, n = -1, 1, 4  # Multiple of 2
        h = (b - a) / (2**n - 1)
        vector_quad = (3 * h / 8) * np.array(
            [1, 3, 3, 2, 3, 3, 2, 3, 3, 2, 3, 3, 2, 3, 3, 1]
        )
        mps_quad = mps_simpson(a, b, n)
        self.assertSimilar(vector_quad, mps_quad.to_vector())

    def test_mps_fifth_order(self):
        a, b, n = -1, 1, 4  # Multiple of 4
        h = (b - a) / (2**n - 1)
        vector_quad = (5 * h / 288) * np.array(
            [19, 75, 50, 50, 75, 38, 75, 50, 50, 75, 38, 75, 50, 50, 75, 19]
        )
        mps_quad = mps_fifth_order(a, b, n)
        self.assertSimilar(vector_quad, mps_quad.to_vector())

    def test_mps_fejer(self):
        a, b, n = -1, 1, 3
        h = (b - a) / 2
        # Implement Fejér vector with iFFT
        N = 2**n
        v = np.zeros(N, dtype=complex)
        for k in range(N // 2):
            v[k] = 2 / (1 - 4 * k**2) * np.exp(1j * k * np.pi / N)
        for k in range(1, N // 2 + 1):
            v[-k] = np.conjugate(v[k])
        if N % 2 == 0:
            v[N // 2] = 0
        vector_quad = h * ifft(v).flatten().real
        mps_quad = mps_fejer(a, b, n)
        self.assertSimilar(vector_quad, mps_quad.to_vector())
