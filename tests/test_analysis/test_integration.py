import numpy as np
from scipy.fft import ifft

from seemps.state import MPS, scprod, mps_tensor_product
from seemps.analysis.factories import mps_exponential
from seemps.analysis.mesh import (
    Mesh,
    ChebyshevInterval,
    QuantizedInterval,
    mps_to_mesh_matrix,
    interleaving_permutation,
)
from seemps.analysis.integration import mesh_to_quadrature_mesh, quadrature_mesh_to_mps
from seemps.analysis.integration.mps_quadratures import (
    mps_clenshaw_curtis,
    mps_trapezoidal,
    mps_simpson38,
    mps_fifth_order,
    mps_fejer,
)
from seemps.analysis.integration.integration import integrate_mps
from seemps.analysis.mesh import ArrayInterval, RegularInterval

from ..tools import SeeMPSTestCase


class TestMPSQuadratures(SeeMPSTestCase):
    def test_mps_trapezoidal(self):
        a, b, n = -1, 1, 3
        h = (b - a) / (2**n - 1)
        vector_quad = (h / 2) * np.array([1, 2, 2, 2, 2, 2, 2, 1])
        mps_quad = mps_trapezoidal(-1, 1, n)
        self.assertSimilar(vector_quad, mps_quad.to_vector())

    def test_mps_simpson38(self):
        a, b, n = -1, 1, 4  # Multiple of 2
        h = (b - a) / (2**n - 1)
        vector_quad = (3 * h / 8) * np.array(
            [1, 3, 3, 2, 3, 3, 2, 3, 3, 2, 3, 3, 2, 3, 3, 1]
        )
        mps_quad = mps_simpson38(a, b, n)
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


class TestMPSIntegration(SeeMPSTestCase):
    def setUp(self):
        self.a, self.b = -2, 2
        self.func = lambda x: np.exp(x)
        self.integral = self.func(self.b) - self.func(self.a)
        self.step = lambda n: (self.b - self.a) / (2**n - 1)

    def test_trapezoidal_integral(self):
        n = 11
        interval = QuantizedInterval(self.a, self.b, n, endpoint_right=True)
        f_mps = mps_exponential(interval)
        q_mps = mps_trapezoidal(interval[0], interval[-1], n)
        integral = scprod(f_mps, q_mps)
        self.assertAlmostEqual(self.integral, integral, places=5)

    def test_simpson38_integral(self):
        n = 10
        interval = QuantizedInterval(self.a, self.b, n, endpoint_right=True)
        f_mps = mps_exponential(interval)
        q_mps = mps_simpson38(interval[0], interval[-1], n)
        integral = scprod(f_mps, q_mps)
        self.assertAlmostEqual(self.integral, integral)

    def test_fifth_order_integral(self):
        n = 8
        interval = QuantizedInterval(self.a, self.b, n, endpoint_right=True)
        f_mps = mps_exponential(interval)
        q_mps = mps_fifth_order(interval[0], interval[-1], n)
        integral = scprod(f_mps, q_mps)
        self.assertAlmostEqual(self.integral, integral)

    def test_fejer_integral(self):
        n = 4
        interval = ChebyshevInterval(self.a, self.b, 2**n)
        f = self.func(interval.to_vector())
        f_mps = MPS.from_vector(f, [2] * n, normalize=False)
        q_mps = mps_fejer(self.a, self.b, n)
        integral = scprod(f_mps, q_mps)
        self.assertAlmostEqual(self.integral, integral)

    def test_multivariate_integral_order_A(self):
        n = 4
        integral_2d = self.integral**2

        # Use Fejér integration
        interval = ChebyshevInterval(self.a, self.b, 2**n)
        f_mps_1d = MPS.from_vector(
            self.func(interval.to_vector()), [2] * n, normalize=False
        )
        f_mps_2d = mps_tensor_product([f_mps_1d, f_mps_1d], mps_order="A")

        mesh = Mesh([interval, interval])
        q_mesh = mesh_to_quadrature_mesh(mesh)

        # Construct the quadrature MPS with TCI
        map_matrix = mps_to_mesh_matrix([n, n])
        physical_dimensions = [2] * (2 * n)
        q_mps_2d = quadrature_mesh_to_mps(q_mesh, map_matrix, physical_dimensions)

        integral = scprod(f_mps_2d, q_mps_2d)
        self.assertAlmostEqual(integral_2d, integral)


class TestIntegrationAPI(SeeMPSTestCase):
    def setUp(self):
        self.a, self.b = -2, 2
        self.func = lambda x: np.exp(x)
        self.integral = self.func(self.b) - self.func(self.a)

    def test_integrate_mps_regular_interval_uses_newton_cotes_rules(self):
        interval = RegularInterval(-1.0, 1.0, 8, endpoint_right=True)
        values = np.exp(interval.to_vector())
        mps = MPS.from_vector(values, [2] * 3, normalize=False)
        expected = scprod(mps, mps_trapezoidal(interval[0], interval[-1], 3))
        self.assertAlmostEqual(integrate_mps(mps, interval), expected)

        interval = RegularInterval(-1.0, 1.0, 16, endpoint_right=True)
        values = np.exp(interval.to_vector())
        mps = MPS.from_vector(values, [2] * 4, normalize=False)
        expected = scprod(mps, mps_fifth_order(interval[0], interval[-1], 4))
        self.assertAlmostEqual(integrate_mps(mps, interval), expected)

    def test_integrate_mps_chebyshev_intervals_support_fejer_and_clenshaw_curtis(self):
        interval = ChebyshevInterval(-1.0, 1.0, 8)
        values = np.exp(interval.to_vector())
        mps = MPS.from_vector(values, [2] * 3, normalize=False)
        expected = scprod(mps, mps_fejer(interval.start, interval.stop, 3))
        self.assertAlmostEqual(integrate_mps(mps, interval), expected)

        interval = ChebyshevInterval(-1.0, 1.0, 4, endpoints=True)
        values = np.exp(interval.to_vector())
        mps = MPS.from_vector(values, [2] * 2, normalize=False)
        expected = scprod(mps, mps_clenshaw_curtis(interval.start, interval.stop, 2))
        self.assertAlmostEqual(integrate_mps(mps, interval), expected)

    def test_integrate_mps_rejects_nonquadrature_intervals(self):
        interval = ArrayInterval(np.array([0.0, 1.0, 2.0, 3.0]))
        mps = MPS.from_vector(interval.to_vector(), [2, 2], normalize=False)
        with self.assertRaises(ValueError):
            integrate_mps(mps, interval)

    def test_mesh_to_quadrature_mesh_regular_interval_and_invalid_interval(self):
        mesh = Mesh([RegularInterval(-1.0, 1.0, 5, endpoint_right=True)])
        q_mesh = mesh_to_quadrature_mesh(mesh)
        self.assertSimilar(
            q_mesh.intervals[0].to_vector(), np.array([7, 32, 12, 32, 7]) / 45.0
        )

        with self.assertRaises(ValueError):
            mesh_to_quadrature_mesh(Mesh([ArrayInterval([1.0, 2.0, 3.0])]))

    def test_multivariate_integral_order_B(self):
        n = 4
        integral_2d = self.integral**2

        # Use Fejér integration
        interval = ChebyshevInterval(self.a, self.b, 2**n)
        f_mps_1d = MPS.from_vector(
            self.func(interval.to_vector()), [2] * n, normalize=False
        )
        f_mps_2d = mps_tensor_product([f_mps_1d, f_mps_1d], mps_order="B")

        mesh = Mesh([interval, interval])
        q_mesh = mesh_to_quadrature_mesh(mesh)

        # Construct the quadrature MPS with TCI
        permutation = interleaving_permutation([n, n])
        map_matrix = mps_to_mesh_matrix([n, n], permutation)
        physical_dimensions = [2] * (2 * n)
        q_mps_2d = quadrature_mesh_to_mps(q_mesh, map_matrix, physical_dimensions)

        integral = scprod(f_mps_2d, q_mps_2d)
        self.assertAlmostEqual(integral_2d, integral)
