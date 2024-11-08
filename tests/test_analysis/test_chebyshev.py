import numpy as np
from numpy.polynomial import Chebyshev
from scipy.special import erf

from seemps.state import MPS, NO_TRUNCATION, DEFAULT_STRATEGY
from seemps.analysis.mesh import RegularInterval
from seemps.analysis.factories import mps_tensor_sum, mps_interval
from seemps.analysis.chebyshev import (
    interpolation_coefficients,
    projection_coefficients,
    cheb2mps,
    estimate_order,
    cheb2mpo,
)
from seemps.analysis.operators import x_mpo

from ..tools import TestCase


class TestChebyshevCoefficients(TestCase):
    def test_interpolation_coefficients_exponential(self):
        f = lambda x: np.exp(x)
        cheb_coeffs = interpolation_coefficients(f, 15, -1, 1)
        correct_coeffs = [
            1.266065877752008,
            1.130318207984970,
            0.271495339534077,
            0.044336849848664,
            0.005474240442094,
            0.000542926311914,
            0.000044977322954,
            0.000003198436462,
            0.000000199212481,
            0.000000011036772,
            0.000000000550590,
            0.000000000024980,
            0.000000000001039,
            0.000000000000040,
            0.000000000000001,
        ]
        assert np.allclose(list(cheb_coeffs), correct_coeffs)

    def test_estimate_order(self):
        """Assert that the estimated coefficients and accuracy in norm-inf are below a tolerance."""
        tolerance = 1e-12
        f = lambda x: np.exp(-(x**2))
        proj_coeffs = projection_coefficients(
            f, order=estimate_order(f, tolerance=tolerance)
        )
        n = 6
        domain = RegularInterval(-1, 1, 2**n)
        mps = cheb2mps(proj_coeffs, domain=domain, strategy=NO_TRUNCATION)
        y_vec = f(domain.to_vector())
        y_mps = mps.to_vector()
        assert proj_coeffs.coef[-1] <= tolerance
        assert max(abs(y_mps - y_vec)) <= tolerance

    def assertSimilarSeries(self, s1, s2, tol=1e-15):
        """Ensure two Chebyshev series are close up to tolerance."""
        if s1.has_sametype(s2) and s1.has_samedomain(s2):
            # We may need to pad series to have the same sizes
            c1 = s1.coef
            c2 = s2.coef
            L = max(c1.size, c2.size)
            if L > c1.size:
                c1 = np.pad(c1, (0, L - c1.size))
            if L > c2.size:
                c2 = np.pad(c2, (0, L - c2.size))
            if np.allclose(c1, c2, tol):
                return
        raise self.failureException(f"Not similar series:\nA={s1}\nB={s2}")

    def test_chebyshev_coefficients_T0(self):
        # At low orders, the three types of coefficients (zeros, extrema and projections) are almost similar
        T0 = Chebyshev([1, 0, 0, 0])
        self.assertSimilarSeries(interpolation_coefficients(T0, 4, -1, 1), T0)
        self.assertSimilarSeries(
            interpolation_coefficients(T0, 4, -1, 1, interpolated_nodes="extrema"),
            T0,
        )
        self.assertSimilarSeries(projection_coefficients(T0, 4, -1, 1), T0)

    def test_chebyshev_coefficients_T1(self):
        T1 = Chebyshev([0, 1, 0, 0])
        self.assertSimilarSeries(interpolation_coefficients(T1, 4, -1, 1), T1)
        self.assertSimilarSeries(
            interpolation_coefficients(T1, 4, -1, 1, interpolated_nodes="extrema"),
            T1,
        )
        self.assertSimilarSeries(projection_coefficients(T1, 4, -1, 1), T1)

    def test_chebyshev_coefficients_T4(self):
        T4 = Chebyshev([0, 0, 0, 0, 1])
        self.assertSimilarSeries(interpolation_coefficients(T4, 5, -1, 1), T4)
        self.assertSimilarSeries(
            interpolation_coefficients(T4, 5 + 1, -1, 1, interpolated_nodes="extrema"),
            T4,  # The extrema are computed for a polynomial of degree d-1 so we need d+1
        )
        self.assertSimilarSeries(projection_coefficients(T4, 5, -1, 1), T4)

    def test_chebyshev_coefficients_T0_other_domain(self):
        T0 = Chebyshev([1, 0, 0, 0], domain=(-2, 4))
        self.assertSimilarSeries(interpolation_coefficients(T0, 4, -2, 4), T0)
        self.assertSimilarSeries(
            interpolation_coefficients(T0, 4, -2, 4, interpolated_nodes="extrema"),
            T0,
        )
        self.assertSimilarSeries(projection_coefficients(T0, 4, -2, 4), T0)

    def test_chebyshev_coefficients_T1_other_domain(self):
        T1 = Chebyshev([0, 1, 0, 0], domain=(-2, 4))
        self.assertSimilarSeries(interpolation_coefficients(T1, 4, -2, 4), T1)
        self.assertSimilarSeries(
            interpolation_coefficients(T1, 4, -2, 4, interpolated_nodes="extrema"),
            T1,
        )
        self.assertSimilarSeries(projection_coefficients(T1, 4, -2, 4), T1)

    def test_chebyshev_coefficients_T4_other_domain(self):
        T4 = Chebyshev([0, 0, 0, 0, 1], domain=(-2, 4))
        self.assertSimilarSeries(interpolation_coefficients(T4, 5, -2, 4), T4)
        self.assertSimilarSeries(
            interpolation_coefficients(T4, 5 + 1, -2, 4, interpolated_nodes="extrema"),
            T4,
        )
        self.assertSimilarSeries(projection_coefficients(T4, 5, -2, 4), T4)

    def test_chebyshev_coefficients_gaussian_derivative(self):
        f = lambda x: np.exp(-x * x)
        df = lambda x: -2 * x * np.exp(-x * x)
        self.assertSimilarSeries(
            interpolation_coefficients(f, 22, -1, 2).deriv(),
            interpolation_coefficients(df, 22, -1, 2),
        )
        self.assertSimilarSeries(
            interpolation_coefficients(
                f, 22, -1, 2, interpolated_nodes="extrema"
            ).deriv(),
            interpolation_coefficients(df, 22, -1, 2, interpolated_nodes="extrema"),
        )
        self.assertSimilarSeries(
            projection_coefficients(f, 22, -1, 2).deriv(),
            projection_coefficients(df, 22, -1, 2),
        )

    def test_chebyshev_coefficients_gaussian_integral(self):
        start = -1
        stop = 2
        f = lambda x: np.exp(-x * x)
        f_intg = lambda x: (np.sqrt(np.pi) / 2) * (erf(x) - erf(start))
        self.assertSimilarSeries(
            interpolation_coefficients(f, 22, start, stop).integ(1, lbnd=start),
            interpolation_coefficients(f_intg, 22, start, stop),
        )
        self.assertSimilarSeries(
            interpolation_coefficients(
                f, 22, start, stop, interpolated_nodes="extrema"
            ).integ(1, lbnd=start),
            interpolation_coefficients(
                f_intg, 22, start, stop, interpolated_nodes="extrema"
            ),
        )
        self.assertSimilarSeries(
            projection_coefficients(f, 22, start, stop).integ(1, lbnd=start),
            projection_coefficients(f_intg, 22, start, stop),
        )


class TestChebyshevMPS(TestCase):
    def test_gaussian_1d(self):
        f = lambda x: np.exp(-(x**2))
        interval = RegularInterval(-1, 2, 2**5)
        mps_cheb_clen = cheb2mps(
            interpolation_coefficients(f, 30, domain=interval),
            domain=interval,
            clenshaw=True,
        )
        mps_cheb_poly = cheb2mps(
            interpolation_coefficients(f, 30, domain=interval),
            domain=interval,
            clenshaw=False,
        )
        self.assertSimilar(f(interval.to_vector()), mps_cheb_clen.to_vector())
        self.assertSimilar(f(interval.to_vector()), mps_cheb_poly.to_vector())

    def test_gaussian_derivative_1d(self):
        f = lambda x: np.exp(-(x**2))
        f_diff = lambda x: -2 * x * np.exp(-(x**2))
        interval = RegularInterval(-1, 2, 2**5)
        c = interpolation_coefficients(f, 30, domain=interval)
        mps_cheb_clen = cheb2mps(c.deriv(1), domain=interval, clenshaw=True)
        mps_cheb_poly = cheb2mps(c.deriv(1), domain=interval, clenshaw=False)
        self.assertSimilar(f_diff(interval.to_vector()), mps_cheb_clen.to_vector())
        self.assertSimilar(f_diff(interval.to_vector()), mps_cheb_poly.to_vector())

    def test_gaussian_integral_1d(self):
        f_intg = lambda x: (np.sqrt(np.pi) / 2) * (erf(x) - erf(-1))
        interval = RegularInterval(-1, 2, 2**5)
        c = interpolation_coefficients(f_intg, 30, domain=interval)
        mps_cheb_clen = cheb2mps(c, domain=interval, clenshaw=True)
        mps_cheb_poly = cheb2mps(c, domain=interval, clenshaw=False)
        self.assertSimilar(f_intg(interval.to_vector()), mps_cheb_clen.to_vector())
        self.assertSimilar(f_intg(interval.to_vector()), mps_cheb_poly.to_vector())

    def test_gaussian_integral_1d_b(self):
        f = lambda x: np.exp(-(x**2))
        f_intg = lambda x: (np.sqrt(np.pi) / 2) * (erf(x) - erf(-1))
        interval = RegularInterval(-1, 2, 2**5)
        c = interpolation_coefficients(f, 30, domain=interval)
        mps_cheb_clen = cheb2mps(
            c.integ(1, lbnd=interval.start), domain=interval, clenshaw=True
        )
        mps_cheb_poly = cheb2mps(
            c.integ(1, lbnd=interval.start), domain=interval, clenshaw=False
        )
        self.assertSimilar(f_intg(interval.to_vector()), mps_cheb_clen.to_vector())
        self.assertSimilar(f_intg(interval.to_vector()), mps_cheb_poly.to_vector())

    def test_gaussian_2d(self):
        f = lambda z: np.exp(-(z**2))
        c = interpolation_coefficients(f, 30, -1, 5)
        sites = 6
        interval_x = RegularInterval(-0.5, 2, 2**sites)
        interval_y = RegularInterval(-0.5, 3, 2**sites)
        mps_x_plus_y = mps_tensor_sum(
            [mps_interval(interval_y), mps_interval(interval_x)]  # type: ignore
        )
        tol = 1e-10
        strategy = DEFAULT_STRATEGY.replace(
            tolerance=tol**2, simplification_tolerance=tol**2
        )
        mps_cheb_clen = cheb2mps(
            c, initial_mps=mps_x_plus_y, strategy=strategy, clenshaw=True
        )
        mps_cheb_poly = cheb2mps(
            c, initial_mps=mps_x_plus_y, strategy=strategy, clenshaw=False
        )
        X, Y = np.meshgrid(interval_x.to_vector(), interval_y.to_vector())
        Z_vector = f(X + Y)
        Z_mps_clen = mps_cheb_clen.to_vector().reshape([2**sites, 2**sites])
        Z_mps_poly = mps_cheb_poly.to_vector().reshape([2**sites, 2**sites])
        self.assertSimilar(Z_vector, Z_mps_clen)
        self.assertSimilar(Z_vector, Z_mps_poly)


class TestChebyshevMPO(TestCase):
    def test_gaussian_mpo(self):
        """
        Tests the interpolation of a diagonal MPO representing a gaussian.
        """
        a, b, n = -1, 1, 5
        dx = (b - a) / 2**n
        x = np.linspace(a, b, 2**n, endpoint=False)

        f = lambda x: np.sin(-(x**2))
        coefficients = interpolation_coefficients(f)

        I = MPS([np.ones((1, 2, 1))] * n)
        mpo_x = x_mpo(n, a, dx)
        mpo_gaussian_clen = cheb2mpo(coefficients, mpo_x, clenshaw=True)
        mpo_gaussian_poly = cheb2mpo(coefficients, mpo_x, clenshaw=False)
        self.assertSimilar(f(x), mpo_gaussian_clen.apply(I).to_vector())
        self.assertSimilar(f(x), mpo_gaussian_poly.apply(I).to_vector())
