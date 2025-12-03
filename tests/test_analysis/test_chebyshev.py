import numpy as np
from numpy.polynomial import Chebyshev
from scipy.special import erf

from seemps.state import MPS, NO_TRUNCATION, DEFAULT_STRATEGY
from seemps.analysis.mesh import RegularInterval
from seemps.analysis.factories import mps_tensor_sum, mps_interval
from seemps.analysis.expansion import (
    ChebyshevExpansion,
    mps_polynomial_expansion,
    mpo_polynomial_expansion,
)
from seemps.analysis.operators import x_mpo

from ..tools import TestCase


class TestChebyshevCoefficients(TestCase):
    def test_interpolation_coefficients_exponential(self):
        f = lambda x: np.exp(x)  # noqa: E731
        expansion = ChebyshevExpansion.interpolate(f, -1, 1, order=15)
        cheb_coeffs = expansion.coeffs
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
        self.assertSimilar(list(cheb_coeffs), correct_coeffs)

    def test_estimate_order(self):
        """Assert that the estimated coefficients and accuracy in norm-inf are below a tolerance."""
        tolerance = 1e-12
        f = lambda x: np.exp(-(x**2))  # noqa: E731
        expansion = ChebyshevExpansion.project(f, -1, 1, order=None)
        proj_coeffs = expansion.coeffs
        n = 6
        domain = RegularInterval(-1, 1, 2**n)
        mps = mps_polynomial_expansion(
            expansion, initial=domain, strategy=NO_TRUNCATION
        )
        y_vec = f(domain.to_vector())
        y_mps = mps.to_vector()
        self.assertTrue(proj_coeffs[-1] <= tolerance)
        self.assertSimilar(y_mps, y_vec, atol=tolerance)

    # TODO: Refactor these tests

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
        # fmt: off
        coeffs_zeros = Chebyshev(ChebyshevExpansion.interpolate(T0, -1, 1, 4, nodes="zeros").coeffs)
        coeffs_extrema = Chebyshev(ChebyshevExpansion.interpolate(T0, -1, 1, 4, nodes="extrema").coeffs)
        coeffs_proj = Chebyshev(ChebyshevExpansion.project(T0, -1, 1, 4).coeffs)
        # fmt: on
        self.assertSimilarSeries(coeffs_zeros, T0)
        self.assertSimilarSeries(coeffs_extrema, T0)
        self.assertSimilarSeries(coeffs_proj, T0)

    def test_chebyshev_coefficients_T1(self):
        T1 = Chebyshev([0, 1, 0, 0])
        # fmt: off
        coeffs_zeros = Chebyshev(ChebyshevExpansion.interpolate(T1, -1, 1, 4, nodes="zeros").coeffs)
        coeffs_extrema = Chebyshev(ChebyshevExpansion.interpolate(T1, -1, 1, 4, nodes="extrema").coeffs)
        coeffs_proj = Chebyshev(ChebyshevExpansion.project(T1, -1, 1, 4).coeffs)
        # fmt: on
        self.assertSimilarSeries(coeffs_zeros, T1)
        self.assertSimilarSeries(coeffs_extrema, T1)
        self.assertSimilarSeries(coeffs_proj, T1)

    def test_chebyshev_coefficients_T4(self):
        T4 = Chebyshev([0, 0, 0, 0, 1])
        # fmt: off
        # The extrema are computed for a polynomial of degree d-1 so we need d+1
        coeffs_zeros = Chebyshev(ChebyshevExpansion.interpolate(T4, -1, 1, 5, nodes="zeros").coeffs)
        coeffs_extrema = Chebyshev(ChebyshevExpansion.interpolate(T4, -1, 1, 5+1, nodes="extrema").coeffs)
        coeffs_proj = Chebyshev(ChebyshevExpansion.project(T4, -1, 1, 5).coeffs)
        # fmt: on
        self.assertSimilarSeries(coeffs_zeros, T4)

        self.assertSimilarSeries(coeffs_extrema, T4)
        self.assertSimilarSeries(coeffs_proj, T4)

    def test_chebyshev_coefficients_T0_other_domain(self):
        T0 = Chebyshev([1, 0, 0, 0], domain=(-2, 4))
        # fmt: off
        coeffs_zeros = Chebyshev(ChebyshevExpansion.interpolate(T0, -2, 4, 4, nodes="zeros").coeffs, domain=(-2,4))
        coeffs_extrema = Chebyshev(ChebyshevExpansion.interpolate(T0, -2, 4, 4, nodes="extrema").coeffs, domain=(-2,4))
        coeffs_proj = Chebyshev(ChebyshevExpansion.project(T0, -2, 4, 4).coeffs, domain=(-2,4))
        # fmt: on
        self.assertSimilarSeries(coeffs_zeros, T0)
        self.assertSimilarSeries(coeffs_extrema, T0)
        self.assertSimilarSeries(coeffs_proj, T0)

    def test_chebyshev_coefficients_T1_other_domain(self):
        T1 = Chebyshev([0, 1, 0, 0], domain=(-2, 4))
        # fmt: off
        coeffs_zeros = Chebyshev(ChebyshevExpansion.interpolate(T1, -2, 4, 4, nodes="zeros").coeffs, domain=(-2,4))
        coeffs_extrema = Chebyshev(ChebyshevExpansion.interpolate(T1, -2, 4, 4, nodes="extrema").coeffs, domain=(-2,4))
        coeffs_proj = Chebyshev(ChebyshevExpansion.project(T1, -2, 4, 4).coeffs, domain=(-2,4))
        # fmt: on
        self.assertSimilarSeries(coeffs_zeros, T1)
        self.assertSimilarSeries(coeffs_extrema, T1)
        self.assertSimilarSeries(coeffs_proj, T1)

    def test_chebyshev_coefficients_T4_other_domain(self):
        T4 = Chebyshev([0, 0, 0, 0, 1], domain=(-2, 4))
        # fmt: off
        coeffs_zeros = Chebyshev(ChebyshevExpansion.interpolate(T4, -2, 4, 5, nodes="zeros").coeffs, domain=(-2,4))
        coeffs_extrema = Chebyshev(ChebyshevExpansion.interpolate(T4, -2, 4, 5+1, nodes="extrema").coeffs, domain=(-2,4))
        coeffs_proj = Chebyshev(ChebyshevExpansion.project(T4, -2, 4, 5).coeffs, domain=(-2,4))
        # fmt: on
        self.assertSimilarSeries(coeffs_zeros, T4)
        self.assertSimilarSeries(coeffs_extrema, T4)
        self.assertSimilarSeries(coeffs_proj, T4)

    def test_chebyshev_coefficients_gaussian_derivative(self):
        f = lambda x: np.exp(-x * x)  # noqa: E731
        df = lambda x: -2 * x * np.exp(-x * x)  # noqa: E731
        # fmt: off
        coeffs_zeros_f = Chebyshev(ChebyshevExpansion.interpolate(f, -1, 2, 22).deriv().coeffs)
        coeffs_zeros_df = Chebyshev(ChebyshevExpansion.interpolate(df, -1, 2, 22).coeffs)
        coeffs_extrema_f = Chebyshev(ChebyshevExpansion.interpolate(f, -1, 2, 22, nodes="extrema").deriv().coeffs)
        coeffs_extrema_df = Chebyshev(ChebyshevExpansion.interpolate(df, -1, 2, 22, nodes="extrema").coeffs)
        coeffs_proj_f = Chebyshev(ChebyshevExpansion.project(f, -1, 2, 22).deriv().coeffs)
        coeffs_proj_df = Chebyshev(ChebyshevExpansion.project(df, -1, 2, 22).coeffs)
        # fmt: on
        self.assertSimilarSeries(coeffs_zeros_f, coeffs_zeros_df)
        self.assertSimilarSeries(coeffs_extrema_f, coeffs_extrema_df)
        self.assertSimilarSeries(coeffs_proj_f, coeffs_proj_df)

    def test_chebyshev_coefficients_gaussian_integral(self):
        start, stop = -1, 2
        f = lambda x: np.exp(-x * x)  # noqa: E731
        F = lambda x: (np.sqrt(np.pi) / 2) * (erf(x) - erf(start))  # noqa: E731
        # fmt: off
        coeffs_zeros_f = Chebyshev(ChebyshevExpansion.interpolate(f, -1, 2, 22).integ(lbnd=start).coeffs)
        coeffs_zeros_F = Chebyshev(ChebyshevExpansion.interpolate(F, -1, 2, 22).coeffs)
        coeffs_extrema_f = Chebyshev(ChebyshevExpansion.interpolate(f, -1, 2, 22, nodes="extrema").integ(lbnd=start).coeffs)
        coeffs_extrema_F = Chebyshev(ChebyshevExpansion.interpolate(F, -1, 2, 22, nodes="extrema").coeffs)
        coeffs_proj_f = Chebyshev(ChebyshevExpansion.project(f, -1, 2, 22).integ(lbnd=start).coeffs)
        coeffs_proj_F = Chebyshev(ChebyshevExpansion.project(F, -1, 2, 22).coeffs)
        # fmt: on

        self.assertSimilarSeries(coeffs_zeros_f, coeffs_zeros_F)
        self.assertSimilarSeries(coeffs_extrema_f, coeffs_extrema_F)
        self.assertSimilarSeries(coeffs_proj_f, coeffs_proj_F)


class TestChebyshevMPS(TestCase):
    def test_gaussian_1d(self):
        f = lambda x: np.exp(-(x**2))  # noqa: E731
        interval = RegularInterval(-1, 2, 2**5)
        mps_cheb_clen = mps_polynomial_expansion(
            ChebyshevExpansion.interpolate(f, -1, 2, 30),
            initial=interval,
            clenshaw=True,
        )
        mps_cheb_poly = mps_polynomial_expansion(
            ChebyshevExpansion.interpolate(f, -1, 2, 30),
            initial=interval,
            clenshaw=False,
        )
        self.assertSimilar(f(interval.to_vector()), mps_cheb_clen.to_vector())
        self.assertSimilar(f(interval.to_vector()), mps_cheb_poly.to_vector())

    def test_gaussian_derivative_1d(self):
        f = lambda x: np.exp(-(x**2))  # noqa: E731
        df = lambda x: -2 * x * np.exp(-(x**2))  # noqa: E731
        interval = RegularInterval(-1, 2, 2**5)
        mps_cheb_clen = mps_polynomial_expansion(
            ChebyshevExpansion.interpolate(f, -1, 2, 30).deriv(1),
            initial=interval,
            clenshaw=True,
        )
        mps_cheb_poly = mps_polynomial_expansion(
            ChebyshevExpansion.interpolate(f, -1, 2, 30).deriv(1),
            initial=interval,
            clenshaw=False,
        )
        self.assertSimilar(df(interval.to_vector()), mps_cheb_clen.to_vector())
        self.assertSimilar(df(interval.to_vector()), mps_cheb_poly.to_vector())

    def test_gaussian_integral_1d(self):
        F = lambda x: (np.sqrt(np.pi) / 2) * (erf(x) - erf(-1))  # noqa: E731
        interval = RegularInterval(-1, 2, 2**5)
        mps_cheb_clen = mps_polynomial_expansion(
            ChebyshevExpansion.interpolate(F, -1, 2, 30),
            initial=interval,
            clenshaw=True,
        )
        mps_cheb_poly = mps_polynomial_expansion(
            ChebyshevExpansion.interpolate(F, -1, 2, 30),
            initial=interval,
            clenshaw=False,
        )
        self.assertSimilar(F(interval.to_vector()), mps_cheb_clen.to_vector())
        self.assertSimilar(F(interval.to_vector()), mps_cheb_poly.to_vector())

    def test_gaussian_integral_1d_b(self):
        f = lambda x: np.exp(-(x**2))  # noqa: E731
        F = lambda x: (np.sqrt(np.pi) / 2) * (erf(x) - erf(-1))  # noqa: E731
        interval = RegularInterval(-1, 2, 2**5)
        # c = interpolation_coefficients(f, 30, domain=interval)
        mps_cheb_clen = mps_polynomial_expansion(
            ChebyshevExpansion.interpolate(f, -1, 2, 30).integ(1, lbnd=interval.start),
            initial=interval,
            clenshaw=True,
        )
        mps_cheb_poly = mps_polynomial_expansion(
            ChebyshevExpansion.interpolate(f, -1, 2, 30).integ(1, lbnd=interval.start),
            initial=interval,
            clenshaw=False,
        )
        self.assertSimilar(F(interval.to_vector()), mps_cheb_clen.to_vector())
        self.assertSimilar(F(interval.to_vector()), mps_cheb_poly.to_vector())

    def test_gaussian_2d(self):
        f = lambda z: np.exp(-(z**2))  # noqa: E731
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
        mps_cheb_clen = mps_polynomial_expansion(
            ChebyshevExpansion.interpolate(f, -1, 5, 30),
            initial=mps_x_plus_y,
            strategy=strategy,
            clenshaw=True,
        )
        mps_cheb_poly = mps_polynomial_expansion(
            ChebyshevExpansion.interpolate(f, -1, 5, 30),
            initial=mps_x_plus_y,
            strategy=strategy,
            clenshaw=False,
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

        f = lambda x: np.sin(-(x**2))  # noqa: E731

        I = MPS([np.ones((1, 2, 1))] * n)
        mpo_x = x_mpo(n, a, dx)
        mpo_gaussian_clen = mpo_polynomial_expansion(
            ChebyshevExpansion.interpolate(f, -1, 1, 30), mpo_x, clenshaw=True
        )
        mpo_gaussian_poly = mpo_polynomial_expansion(
            ChebyshevExpansion.interpolate(f, -1, 1, 30), mpo_x, clenshaw=False
        )
        self.assertSimilar(f(x), mpo_gaussian_clen.apply(I).to_vector())
        self.assertSimilar(f(x), mpo_gaussian_poly.apply(I).to_vector())
