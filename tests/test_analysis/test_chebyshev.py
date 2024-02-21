import numpy as np
from numpy.polynomial import Chebyshev
from scipy.special import erf

from seemps.analysis.mesh import RegularHalfOpenInterval
from seemps.analysis.factories import mps_tensor_sum, mps_interval
from seemps.analysis.chebyshev import (
    DEFAULT_CHEBYSHEV_STRATEGY,
    chebyshev_coefficients,
    chebyshev_approximation,
    cheb2mps,
)

from ..tools import TestCase


class TestChebyshevExpansion(TestCase):
    def test_chebyshev_coefficients_exponential(self):
        f = lambda x: np.exp(x)
        cheb_coeffs = chebyshev_coefficients(f, 15, -1, 1)
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
        T0 = Chebyshev([1, 0, 0, 0])
        self.assertSimilarSeries(chebyshev_coefficients(T0, 4, -1, 1), T0)

    def test_chebyshev_coefficients_T1(self):
        T1 = Chebyshev([0, 1, 0, 0])
        self.assertSimilarSeries(chebyshev_coefficients(T1, 4, -1, 1), T1)

    def test_chebyshev_coefficients_T4(self):
        T4 = Chebyshev([0, 0, 0, 0, 1])
        self.assertSimilarSeries(chebyshev_coefficients(T4, 5, -1, 1), T4)

    def test_chebyshev_coefficients_T0_other_domain(self):
        T0 = Chebyshev([1, 0, 0, 0], domain=(-2, 4))
        self.assertSimilarSeries(chebyshev_coefficients(T0, 4, -2, 4), T0)

    def test_chebyshev_coefficients_T1_other_domain(self):
        T1 = Chebyshev([0, 1, 0, 0], domain=(-2, 4))
        self.assertSimilarSeries(chebyshev_coefficients(T1, 4, -2, 4), T1)

    def test_chebyshev_coefficients_T4_other_domain(self):
        T4 = Chebyshev([0, 0, 0, 0, 1], domain=(-2, 4))
        self.assertSimilarSeries(chebyshev_coefficients(T4, 5, -2, 4), T4)

    def test_chebyshev_coefficients_gaussian_derivative(self):
        f = lambda x: np.exp(-x * x)
        df = lambda x: -2 * x * np.exp(-x * x)
        self.assertSimilarSeries(
            chebyshev_coefficients(f, 22, -1, 2).deriv(),
            chebyshev_coefficients(df, 22, -1, 2),
        )

    def test_chebyshev_coefficients_gaussian_integral(self):
        start = -1
        stop = 2
        f = lambda x: np.exp(-x * x)
        f_intg = lambda x: (np.sqrt(np.pi) / 2) * (erf(x) - erf(start))
        self.assertSimilarSeries(
            chebyshev_coefficients(f, 22, start, stop).integ(1, lbnd=start),
            chebyshev_coefficients(f_intg, 22, start, stop),
        )


class TestChebyshevMPS(TestCase):
    def test_gaussian_1d(self):
        f = lambda x: np.exp(-(x**2))
        interval = RegularHalfOpenInterval(-1, 2, 2**5)
        mps_cheb = chebyshev_approximation(f, 20, interval)
        self.assertSimilar(f(interval.to_vector()), mps_cheb.to_vector())

    def test_gaussian_derivative_1d(self):
        f = lambda x: np.exp(-(x**2))
        f_diff = lambda x: -2 * x * np.exp(-(x**2))
        interval = RegularHalfOpenInterval(-1, 2, 2**5)
        mps_cheb = chebyshev_approximation(f, 22, interval, differentiation_order=1)
        self.assertSimilar(f_diff(interval.to_vector()), mps_cheb.to_vector())

    def test_gaussian_integral_1d(self):
        f_intg = lambda x: (np.sqrt(np.pi) / 2) * (erf(x) - erf(-1))
        interval = RegularHalfOpenInterval(-1, 2, 2**5)
        mps_cheb = chebyshev_approximation(f_intg, 20, interval)
        self.assertSimilar(f_intg(interval.to_vector()), mps_cheb.to_vector())

    def test_gaussian_integral_1d_b(self):
        f = lambda x: np.exp(-(x**2))
        f_intg = lambda x: (np.sqrt(np.pi) / 2) * (erf(x) - erf(-1))
        interval = RegularHalfOpenInterval(-1, 2, 2**5)
        mps_cheb = chebyshev_approximation(f, 20, interval, differentiation_order=-1)
        self.assertSimilar(f_intg(interval.to_vector()), mps_cheb.to_vector())

    def test_gaussian_2d(self):
        f = lambda z: np.exp(-(z**2))
        c = chebyshev_coefficients(f, 30, -1, 5)
        sites = 6
        interval_x = RegularHalfOpenInterval(-0.5, 2, 2**sites)
        interval_y = RegularHalfOpenInterval(-0.5, 3, 2**sites)
        mps_x_plus_y = mps_tensor_sum(
            [mps_interval(interval_y), mps_interval(interval_x)]
        )
        strategy = DEFAULT_CHEBYSHEV_STRATEGY.replace(
            tolerance=1e-15, simplification_tolerance=1e-15
        )
        mps_cheb = cheb2mps(c, x=mps_x_plus_y, strategy=strategy)
        X, Y = np.meshgrid(interval_x.to_vector(), interval_y.to_vector())
        Z_vector = f(X + Y)
        Z_mps = mps_cheb.to_vector().reshape([2**sites, 2**sites])
        self.assertSimilar(Z_vector, Z_mps)


class TestChebyshevMPO(TestCase):
    pass
