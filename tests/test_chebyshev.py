import numpy as np
from scipy.special import erf

from seemps.analysis import (
    RegularHalfOpenInterval,
    chebyshev_coefficients,
    chebyshev_approximation,
    cheb2mps,
    mps_tensor_sum,
    mps_tensor_product,
    mps_interval,
)
from seemps.analysis.chebyshev import DEFAULT_CHEBYSHEV_STRATEGY
from seemps.state import Strategy
from seemps.operators import MPO
from numpy.polynomial import Polynomial, Chebyshev

from .tools import TestCase


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
        start = 0
        stop = 2
        sites = 5
        order = 20
        f = lambda x: np.exp(-(x**2))
        interval = RegularHalfOpenInterval(start, stop, 2**sites)
        mps_cheb = chebyshev_approximation(f, order, interval)
        self.assertSimilar(f(interval.to_vector()), mps_cheb.to_vector())

    def test_gaussian_derivative_1d(self):
        start = -1
        stop = 2
        sites = 5
        order = 22
        f = lambda x: np.exp(-(x**2))
        f_diff = lambda x: -2 * x * np.exp(-(x**2))
        interval = RegularHalfOpenInterval(start, stop, 2**sites)
        mps_cheb = chebyshev_approximation(f, order, interval, differentiation_order=1)
        self.assertSimilar(f_diff(interval.to_vector()), mps_cheb.to_vector())

    def test_gaussian_integral_1d(self):
        start = -1
        stop = 2
        sites = 5
        order = 20
        f_intg = lambda x: (np.sqrt(np.pi) / 2) * (erf(x) - erf(start))
        interval = RegularHalfOpenInterval(start, stop, 2**sites)
        mps_cheb = chebyshev_approximation(f_intg, order, interval)
        self.assertSimilar(f_intg(interval.to_vector()), mps_cheb.to_vector())

    def test_gaussian_integral_1d_b(self):
        start = -1
        stop = 2
        sites = 5
        order = 20
        f = lambda x: np.exp(-(x**2))
        f_intg = lambda x: (np.sqrt(np.pi) / 2) * (erf(x) - erf(start))
        interval = RegularHalfOpenInterval(start, stop, 2**sites)
        mps_cheb = chebyshev_approximation(f, order, interval, differentiation_order=-1)
        self.assertSimilar(f_intg(interval.to_vector()), mps_cheb.to_vector())

    # TODO: This does not have a place here. Move tests to separate file
    def test_tensor_product(self):
        start = -1
        stop = 2
        sites = 5
        interval = RegularHalfOpenInterval(start, stop, 2**sites)
        vector = interval.to_vector()
        mps_intv = mps_interval(interval)
        mps_domain = mps_tensor_product([mps_intv, mps_intv])
        Z_mps = mps_domain.to_vector().reshape((2**sites, 2**sites))
        X, Y = np.meshgrid(vector, vector)
        self.assertSimilar(Z_mps, X * Y)

    # TODO: This does not have a place here. Move tests to separate file
    def test_tensor_sum(self):
        start = -2
        stop = 2
        sites = 5
        interval = RegularHalfOpenInterval(start, stop, 2**sites)
        vector = interval.to_vector()
        mps_intv = mps_interval(interval)
        mps_domain = mps_tensor_sum([mps_intv, mps_intv])
        Z_mps = mps_domain.to_vector().reshape((2**sites, 2**sites))
        X, Y = np.meshgrid(vector, vector)
        self.assertSimilar(Z_mps, X + Y)

    def test_gaussian_2d(self):
        start = -0.5
        stop = 0.5
        sites = 5
        order = 40
        interval = RegularHalfOpenInterval(start, stop, 2**sites)
        vector = interval.to_vector()
        func = lambda x: np.exp(-(x**2))
        strategy = DEFAULT_CHEBYSHEV_STRATEGY.replace(
            tolerance=1e-15, simplification_tolerance=1e-15
        )
        mps_x_plus_y = mps_tensor_sum([mps_interval(interval)] * 2)
        mps_cheb = cheb2mps(
            # Note that we give `chebyshev_coefficients` an interval
            # that covers the smallest and largest values of `X + Y`
            chebyshev_coefficients(func, order, 2 * start, 2 * stop),
            x=mps_x_plus_y,
            strategy=strategy,
        )
        Z_mps = mps_cheb.to_vector().reshape((2**sites, 2**sites))
        X, Y = np.meshgrid(vector, vector)
        Z_vector = func(X + Y)
        self.assertSimilar(Z_mps, Z_vector)


class TestChebyshevMPO(TestCase):
    pass
    # def test_gaussian_on_diagonal_mpo(self):
    #     start = -1
    #     stop = 2
    #     sites = 5
    #     order = 10
    #     f = lambda x: np.exp(-(x**2))
    #     mpo_domain = position_mpo(start, stop, sites)
    #     mpo_cheb = chebyshev_approximation(f, order, mpo_domain)
