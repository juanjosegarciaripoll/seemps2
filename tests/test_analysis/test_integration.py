import numpy as np
from scipy.fft import ifft
from seemps.state import MPS
from seemps.analysis.factories import mps_exponential, mps_tensor_product
from seemps.analysis.mesh import Mesh, RegularInterval, ChebyshevInterval
from seemps.analysis.integration import (
    mps_midpoint,
    mps_trapezoidal,
    mps_simpson,
    mps_fifth_order,
    mps_fejer,
    mps_clenshaw_curtis,
    integrate_mps,
)
from ..tools import TestCase


class TestMPSQuadratures(TestCase):
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
        mps_quad = mps_trapezoidal(-1, 1, n)
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
        a, b, n = -2, 2, 5
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
        vector_quad = h * ifft(v).real  # type: ignore
        mps_quad = mps_fejer(a, b, n)
        self.assertSimilar(vector_quad, mps_quad.to_vector())

    def test_mps_clenshaw_curtis(self):
        a, b, n = -2, 2, 5
        h = (b - a) / 2
        # Implement Clenshaw-Curtis vector with iFFT
        N = int(2**n) - 1
        v = np.zeros(N)
        g = np.zeros(N)
        w0 = 1 / (N**2 - 1 + (N % 2))
        for k in range(N // 2):
            v[k] = 2 / (1 - 4 * k**2)
            g[k] = -w0
        v[N // 2] = (N - 3) / (2 * (N // 2) - 1) - 1
        g[N // 2] = w0 * ((2 - (N % 2)) * N - 1)
        for k in range(1, N // 2 + 1):
            v[-k] = v[k]
            g[-k] = g[k]
        w = np.fft.ifft(v + g).real
        vector_quad = h * np.hstack((w, w[0]))
        mps_quad = mps_clenshaw_curtis(a, b, n)
        self.assertSimilar(vector_quad, mps_quad.to_vector())


class TestMPSIntegrals(TestCase):
    def setUp(self):
        self.a, self.b = -2, 2
        self.func = lambda x: np.exp(x)
        self.integral = self.func(self.b) - self.func(self.a)
        self.step = lambda n: (self.b - self.a) / (2**n - 1)

    def test_trapezoidal_integral(self):
        n = 11
        mps = mps_exponential(self.a, self.b + self.step(n), n)
        interval = RegularInterval(self.a, self.b, 2**n, endpoint_right=True)
        integral = integrate_mps(mps, interval)
        self.assertAlmostEqual(self.integral, integral, places=5)

    def test_simpson_integral(self):
        n = 10
        mps = mps_exponential(self.a, self.b + self.step(n), n)
        interval = RegularInterval(self.a, self.b, 2**n)
        integral = integrate_mps(mps, interval)
        self.assertAlmostEqual(self.integral, integral)

    def test_fifth_order_integral(self):
        n = 8
        mps = mps_exponential(self.a, self.b + self.step(n), n)
        interval = RegularInterval(self.a, self.b, 2**n)
        integral = integrate_mps(mps, interval)
        self.assertAlmostEqual(self.integral, integral)

    def test_fejer_integral(self):
        n = 4
        interval = ChebyshevInterval(self.a, self.b, 2**n)
        mps = MPS.from_vector(self.func(interval.to_vector()), [2] * n, normalize=False)
        integral = integrate_mps(mps, interval)
        self.assertAlmostEqual(self.integral, integral)

    def test_clenshaw_curtis_integral(self):
        n = 4
        interval = ChebyshevInterval(self.a, self.b, 2**n, endpoints=True)
        mps = MPS.from_vector(self.func(interval.to_vector()), [2] * n, normalize=False)
        integral = integrate_mps(mps, interval)
        self.assertAlmostEqual(self.integral, integral)

    def test_multivariate_integral_order_A(self):
        n = 4
        integral_2d = self.integral**2
        # Fejér quadrature in first variable
        interval_fj = ChebyshevInterval(self.a, self.b, 2**n)
        mps_fj = MPS.from_vector(
            self.func(interval_fj.to_vector()), [2] * n, normalize=False
        )
        # CC quadrature in second variable
        interval_cc = ChebyshevInterval(self.a, self.b, 2**n, endpoints=True)
        mps_cc = MPS.from_vector(
            self.func(interval_cc.to_vector()), [2] * n, normalize=False
        )
        mps = mps_tensor_product([mps_fj, mps_cc], mps_order="A")
        mesh = Mesh([interval_fj, interval_cc])
        integral = integrate_mps(mps, mesh, mps_order="A")
        self.assertAlmostEqual(integral_2d, integral)

    def test_multivariate_integral_order_B(self):
        n = 4
        integral_2d = self.integral**2
        # Fejér quadrature in first variable
        interval_fj = ChebyshevInterval(self.a, self.b, 2**n)
        mps_fj = MPS.from_vector(
            self.func(interval_fj.to_vector()), [2] * n, normalize=False
        )
        # CC quadrature in second variable
        interval_cc = ChebyshevInterval(self.a, self.b, 2**n, endpoints=True)
        mps_cc = MPS.from_vector(
            self.func(interval_cc.to_vector()), [2] * n, normalize=False
        )
        mps = mps_tensor_product([mps_fj, mps_cc], mps_order="B")
        mesh = Mesh([interval_fj, interval_cc])
        integral = integrate_mps(mps, mesh, mps_order="B")
        self.assertAlmostEqual(integral_2d, integral)
