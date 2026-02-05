import numpy as np
from numpy.polynomial import Chebyshev
from scipy.special import erf
import seemps
from seemps.state import MPS, DEFAULT_STRATEGY, NO_TRUNCATION, mps_tensor_sum
from seemps.analysis.mesh import RegularInterval
from seemps.analysis.factories import mps_interval
from seemps.analysis.expansion import (
    ChebyshevExpansion,
    LegendreExpansion,
    PowerExpansion,
)
from seemps.analysis.operators import x_mpo
from seemps.typing import Vector

from ..tools import SeeMPSTestCase
from .tools_interpolation import gaussian


class TestChebyshevCoefficients(SeeMPSTestCase):
    def test_expansion_rejects_wrong_literal(self):
        with self.assertRaises(TypeError):
            ChebyshevExpansion.interpolate(np.exp, interpolated_nodes="else")  # type: ignore

    def test_interpolation_coefficients_exponential(self):
        cheb_coeffs = ChebyshevExpansion.interpolate(np.exp, (-1, 1), 15).coefficients
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
        a, b, n = -2, 2, 6
        domain = RegularInterval(-2, 2, 2**n)
        x = domain.to_vector()

        order = ChebyshevExpansion.estimate_order(gaussian, (a, b), tol=tolerance)
        coeffs = ChebyshevExpansion.project(gaussian, (a, b), order).coefficients
        self.assertTrue(coeffs[-1] <= tolerance)

        expansion = ChebyshevExpansion(coeffs, (-2, 2))
        mps = expansion.to_mps(argument=domain, strategy=NO_TRUNCATION)
        y_vec = gaussian(x)
        self.assertSimilar(mps, y_vec, atol=tolerance)

    def test_estimate_order_fails_when_max_order_is_exceeded(self):
        with self.assertRaises(ValueError):
            ChebyshevExpansion.estimate_order(gaussian, (-2, 2), max_order=10)

    def assert_similar_coefficients(
        self,
        coeffs: Vector,
        T1: Chebyshev,
        domain: tuple[float, float] = (-1, 1),
        tol: float = 1e-15,
    ):
        T2 = Chebyshev(coeffs, domain=domain)
        if T1.has_sametype(T2) and T1.has_samedomain(T2):
            c1, c2 = T1.coef, T2.coef
            L = max(c1.size, c2.size)
            if c1.size != L:
                c1 = np.pad(c1, (0, L - c1.size))
            if c2.size != L:
                c2 = np.pad(c2, (0, L - c2.size))
            if np.allclose(c1, c2, tol):
                return
        raise self.failureException(f"Not similar series:\nA={T1}\nB={T2}")

    def assert_expansion_match(self, T: Chebyshev, domain, order, extrema_shift=False):
        (a, b) = domain
        func = lambda x: T(x)  # noqa: E731
        order_e = order + 1 if extrema_shift else order
        zeros = ChebyshevExpansion.interpolate(
            func, (a, b), order, "zeros"
        ).coefficients
        extrema = ChebyshevExpansion.interpolate(
            func, (a, b), order_e, "extrema"
        ).coefficients
        proj = ChebyshevExpansion.project(func, (a, b), order).coefficients
        self.assert_similar_coefficients(zeros, T, domain)
        self.assert_similar_coefficients(extrema, T, domain)
        self.assert_similar_coefficients(proj, T, domain)

    def test_chebyshev_coefficients_T0(self):
        T0 = Chebyshev([1, 0, 0, 0])
        self.assert_expansion_match(T0, domain=(-1, 1), order=4)

    def test_chebyshev_coefficients_T1(self):
        T1 = Chebyshev([0, 1, 0, 0])
        self.assert_expansion_match(T1, domain=(-1, 1), order=4)

    def test_chebyshev_coefficients_T4(self):
        T4 = Chebyshev([0, 0, 0, 0, 1])
        self.assert_expansion_match(T4, domain=(-1, 1), order=5, extrema_shift=True)

    def test_chebyshev_coefficients_T0_other_domain(self):
        T0 = Chebyshev([1, 0, 0, 0], domain=(-2, 4))
        self.assert_expansion_match(T0, domain=(-2, 4), order=4)

    def test_chebyshev_coefficients_T1_other_domain(self):
        T1 = Chebyshev([0, 1, 0, 0], domain=(-2, 4))
        self.assert_expansion_match(T1, domain=(-2, 4), order=4)

    def test_chebyshev_coefficients_T4_other_domain(self):
        T4 = Chebyshev([0, 0, 0, 0, 1], domain=(-2, 4))
        self.assert_expansion_match(T4, domain=(-2, 4), order=5, extrema_shift=True)

    def test_chebyshev_coefficients_gaussian_derivative(self):
        f = lambda x: np.exp(-x * x)  # noqa: E731
        df = lambda x: -2 * x * np.exp(-x * x)  # noqa: E731
        c_f = ChebyshevExpansion.interpolate(f, (-1, 2), 22).deriv().coefficients
        c_df = ChebyshevExpansion.interpolate(df, (-1, 2), 22).coefficients
        self.assertSimilar(c_f, c_df[:-1])
        c_f = (
            ChebyshevExpansion.interpolate(f, (-1, 2), 22, "extrema")
            .deriv()
            .coefficients
        )
        c_df = ChebyshevExpansion.interpolate(df, (-1, 2), 22, "extrema").coefficients
        self.assertSimilar(c_f, c_df[:-1])
        c_f = ChebyshevExpansion.project(f, (-1, 2), 22).deriv().coefficients
        c_df = ChebyshevExpansion.project(df, (-1, 2), 22).coefficients
        self.assertSimilar(c_f, c_df[:-1])

    def test_chebyshev_coefficients_gaussian_integral(self):
        a, b = -1, 2
        order = 22
        f = lambda x: np.exp(-x * x)  # noqa: E731
        F = lambda x: (np.sqrt(np.pi) / 2) * (erf(x) - erf(a))  # noqa: E731
        c_f = (
            ChebyshevExpansion.interpolate(f, (a, b), order).integ(lbnd=a).coefficients
        )
        c_F = ChebyshevExpansion.interpolate(F, (a, b), order).coefficients
        self.assertSimilar(c_f[:-1], c_F)
        c_f = (
            ChebyshevExpansion.interpolate(f, (a, b), order, "extrema")
            .integ(lbnd=a)
            .coefficients[:-1]
        )
        c_F = ChebyshevExpansion.interpolate(F, (a, b), order, "extrema").coefficients
        self.assertSimilar(c_f, c_F)
        c_f = ChebyshevExpansion.project(f, (a, b), order).integ(lbnd=a).coefficients
        c_F = ChebyshevExpansion.project(F, (a, b), order).coefficients
        self.assertSimilar(c_f[:-1], c_F)


class TestChebyshevMPS(SeeMPSTestCase):
    def test_gaussian_1d(self):
        f = lambda x: np.exp(-(x**2))  # noqa: E731
        a, b, n, order = -1, 2, 5, 30
        interval = RegularInterval(a, b, 2**n)
        x = interval.to_vector()

        expansion = ChebyshevExpansion.interpolate(f, (a, b), order)
        mps_cheb_clen = expansion.to_mps(argument=interval, clenshaw=True)
        mps_cheb_poly = expansion.to_mps(argument=interval, clenshaw=False)
        self.assertSimilar(f(x), mps_cheb_clen)
        self.assertSimilar(f(x), mps_cheb_poly)

    def test_gaussian_1d_inside_approximation_domain(self):
        # Test that any argument defined *within* the approximation domain is well approximated.
        f = lambda x: np.exp(-(x**2))  # noqa: E731
        approximation_domain = (-2, 3)
        n, order = 5, 30

        interval = RegularInterval(-1, 1, 2**n)  # Argument does not match domain
        x = interval.to_vector()

        expansion = ChebyshevExpansion.interpolate(f, approximation_domain, order)
        mps_cheb_clen = expansion.to_mps(argument=interval, clenshaw=True)
        mps_cheb_poly = expansion.to_mps(argument=interval, clenshaw=False)
        self.assertSimilar(f(x), mps_cheb_clen)
        self.assertSimilar(f(x), mps_cheb_poly)

    def test_gaussian_derivative_1d(self):
        f = lambda x: np.exp(-(x**2))  # noqa: E731
        df = lambda x: -2 * x * np.exp(-(x**2))  # noqa: E731
        a, b, n, order = -1, 2, 5, 30
        interval = RegularInterval(a, b, 2**n)
        x = interval.to_vector()

        expansion = ChebyshevExpansion.interpolate(f, (a, b), order).deriv(1)
        mps_cheb_clen = expansion.to_mps(argument=interval, clenshaw=True)
        mps_cheb_poly = expansion.to_mps(argument=interval, clenshaw=False)
        self.assertSimilar(df(x), mps_cheb_clen)
        self.assertSimilar(df(x), mps_cheb_poly)

    def test_gaussian_integral_1d(self):
        F = lambda x: (np.sqrt(np.pi) / 2) * (erf(x) - erf(-1))  # noqa: E731
        a, b, n, order = -1, 2, 5, 30
        interval = RegularInterval(a, b, 2**n)
        x = interval.to_vector()

        expansion = ChebyshevExpansion.interpolate(F, (a, b), order)
        mps_cheb_clen = expansion.to_mps(argument=interval, clenshaw=True)
        mps_cheb_poly = expansion.to_mps(argument=interval, clenshaw=False)
        self.assertSimilar(F(x), mps_cheb_clen)
        self.assertSimilar(F(x), mps_cheb_poly)

    def test_gaussian_integral_1d_b(self):
        f = lambda x: np.exp(-(x**2))  # noqa: E731
        F = lambda x: (np.sqrt(np.pi) / 2) * (erf(x) - erf(-1))  # noqa: E731
        a, b, n, order = -1, 2, 5, 30
        interval = RegularInterval(a, b, 2**n)
        x = interval.to_vector()

        expansion = ChebyshevExpansion.interpolate(f, (a, b), order).integ(1, lbnd=a)
        mps_cheb_clen = expansion.to_mps(argument=interval, clenshaw=True)
        mps_cheb_poly = expansion.to_mps(argument=interval, clenshaw=False)
        self.assertSimilar(F(x), mps_cheb_clen)
        self.assertSimilar(F(x), mps_cheb_poly)

    def test_gaussian_2d(self):
        f = lambda z: np.exp(-(z**2))  # noqa: E731

        n = 6
        ix = RegularInterval(-0.5, 2, 2**n)
        iy = RegularInterval(-0.5, 3, 2**n)
        mps_x_plus_y = mps_tensor_sum([mps_interval(iy), mps_interval(ix)])

        strategy = DEFAULT_STRATEGY.replace(tolerance=1e-20)
        expansion = ChebyshevExpansion.interpolate(f, (-1, 5), 30)
        mps_cheb_clen = expansion.to_mps(
            argument=mps_x_plus_y, strategy=strategy, clenshaw=True
        )
        mps_cheb_poly = expansion.to_mps(
            argument=mps_x_plus_y, strategy=strategy, clenshaw=False
        )

        X, Y = np.meshgrid(ix.to_vector(), iy.to_vector())
        Z_ref = f(X + Y)
        Z_clen = mps_cheb_clen.to_vector().reshape([2**n, 2**n])
        Z_poly = mps_cheb_poly.to_vector().reshape([2**n, 2**n])
        self.assertSimilar(Z_ref, Z_clen)
        self.assertSimilar(Z_ref, Z_poly)


class TestChebyshevMPO(SeeMPSTestCase):
    def test_gaussian_mpo(self):
        a, b, n = -1, 1, 5
        dx = (b - a) / 2**n
        x = np.linspace(a, b, 2**n, endpoint=False)
        mpo_x = x_mpo(n, a, dx)

        f = lambda x: np.sin(-(x**2))  # noqa: E731
        expansion = ChebyshevExpansion.interpolate(f, (a, b), order=30)
        mpo_leg_clen = expansion.to_mpo(mpo_x, clenshaw=True)
        mpo_leg_poly = expansion.to_mpo(mpo_x, clenshaw=False)

        I = MPS([np.ones((1, 2, 1))] * n)
        y_clen = mpo_leg_clen.apply(I)
        y_poly = mpo_leg_poly.apply(I)
        self.assertSimilar(f(x), y_clen)
        self.assertSimilar(f(x), y_poly)


# TODO: Refactor tests and combine by PolynomialExpansion


class TestLegendreMPS(SeeMPSTestCase):
    def test_gaussian_1d(self):
        f = lambda x: np.exp(-(x**2))  # noqa: E731
        a, b, n, order = -1, 2, 5, 30
        interval = RegularInterval(a, b, 2**n)
        x = interval.to_vector()

        expansion = LegendreExpansion.project(f, order, (a, b))
        mps_leg_clen = expansion.to_mps(argument=interval, clenshaw=True)
        mps_leg_poly = expansion.to_mps(argument=interval, clenshaw=False)

        self.assertSimilar(f(x), mps_leg_clen)
        self.assertSimilar(f(x), mps_leg_poly)

    def test_gaussian_2d(self):
        f = lambda z: np.exp(-(z**2))  # noqa: E731
        n = 6

        ix = RegularInterval(-0.5, 2, 2**n)
        iy = RegularInterval(-0.5, 3, 2**n)
        mps_x_plus_y = mps_tensor_sum([mps_interval(iy), mps_interval(ix)])

        strategy = DEFAULT_STRATEGY.replace(tolerance=1e-20)
        expansion = LegendreExpansion.project(f, 30, (-1, 5))
        mps_leg_clen = expansion.to_mps(
            argument=mps_x_plus_y,
            strategy=strategy,
            clenshaw=True,
        )
        mps_leg_poly = expansion.to_mps(
            argument=mps_x_plus_y,
            strategy=strategy,
            clenshaw=False,
        )

        X, Y = np.meshgrid(ix.to_vector(), iy.to_vector())
        Z_ref = f(X + Y)
        Z_clen = mps_leg_clen.to_vector().reshape([2**n, 2**n])
        Z_poly = mps_leg_poly.to_vector().reshape([2**n, 2**n])
        self.assertSimilar(Z_ref, Z_clen)
        self.assertSimilar(Z_ref, Z_poly)


class TestLegendreMPO(SeeMPSTestCase):
    def test_gaussian_mpo(self):
        a, b, n = -1, 1, 5
        dx = (b - a) / 2**n
        x = np.linspace(a, b, 2**n, endpoint=False)
        mpo_x = x_mpo(n, a, dx)

        f = lambda x: np.sin(-(x**2))  # noqa: E731
        expansion = LegendreExpansion.project(f, order=30, approximation_domain=(a, b))
        mpo_leg_clen = expansion.to_mpo(mpo_x, clenshaw=True)
        mpo_leg_poly = expansion.to_mpo(mpo_x, clenshaw=False)

        I = MPS([np.ones((1, 2, 1))] * n)
        y_clen = mpo_leg_clen.apply(I)
        y_poly = mpo_leg_poly.apply(I)
        self.assertSimilar(f(x), y_clen)
        self.assertSimilar(f(x), y_poly)


class TestPowerExpansion(SeeMPSTestCase):
    def test_mps_expansion(self):
        a, b, n = -1, 1, 10
        interval = RegularInterval(a, b, 2**n, endpoint_right=True)
        x = interval.to_vector()

        rng = np.random.default_rng(0x22)
        coeffs = rng.normal(loc=0.0, scale=1.0, size=20)
        p = np.polynomial.Polynomial(coeffs)
        y = p(x)

        expansion = PowerExpansion(coeffs)
        mps_clen = expansion.to_mps(argument=interval, clenshaw=True)
        self.assertSimilar(y, mps_clen, atol=1e-6)
        mps_poly = expansion.to_mps(argument=interval, clenshaw=False)
        self.assertSimilar(y, mps_poly, atol=1e-6)

    def test_mpo_expansion(self):
        a, b, n = -1, 1, 10
        interval = RegularInterval(a, b, 2**n, endpoint_right=True)
        x = interval.to_vector()

        rng = np.random.default_rng(0x22)
        coeffs = rng.normal(loc=0.0, scale=1.0, size=20)
        p = np.polynomial.Polynomial(coeffs)
        y = p(x)

        expansion = PowerExpansion(coeffs)
        mps_x = MPS.from_vector(x, [2] * n, normalize=False)
        mpo_x = seemps.operators.projectors.diagonal_mpo_from_mps(mps_x)
        self.assertSimilar(mpo_x, np.diag(x))
        mps_y = MPS.from_vector(y, [2] * n, normalize=False)
        mpo_y = seemps.operators.projectors.diagonal_mpo_from_mps(mps_y)
        self.assertSimilar(mpo_y, np.diag(y))
        mpo_clen = expansion.to_mpo(mpo_x, clenshaw=True)
        self.assertSimilar(mpo_clen, mpo_y, atol=1e-6)
        mpo_poly = expansion.to_mpo(mpo_x, clenshaw=False)
        self.assertSimilar(mpo_poly, mpo_y, atol=1e-6)
