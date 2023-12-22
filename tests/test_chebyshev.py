import numpy as np
from scipy.special import erf

from seemps.analysis import (
    RegularHalfOpenInterval,
    chebyshev_coefficients,
    chebyshev_approximation,
    mps_tensor_sum,
    mps_tensor_product,
    mps_interval,
)
from seemps.state import Strategy
from seemps.operators import MPO

from .tools import TestCase


class TestChebyshevMPS(TestCase):
    def test_chebyshev_coefficients(self):
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
        assert np.allclose(cheb_coeffs, correct_coeffs)

    def test_gaussian_1d(self):
        start = -1
        stop = 2
        sites = 5
        order = 20
        f = lambda x: np.exp(-(x**2))
        interval = RegularHalfOpenInterval(start, stop, 2**sites)
        mps_domain = mps_interval(interval)
        mps_cheb = chebyshev_approximation(f, order, mps_domain)
        self.assertSimilar(f(interval.to_vector()), mps_cheb.to_vector())

    def test_gaussian_derivative_1d(self):
        start = -1
        stop = 2
        sites = 5
        order = 22
        f = lambda x: np.exp(-(x**2))
        f_diff = lambda x: -2 * x * np.exp(-(x**2))
        interval = RegularHalfOpenInterval(start, stop, 2**sites)
        mps_domain = mps_interval(interval)
        mps_cheb = chebyshev_approximation(
            f, order, mps_domain, differentiation_order=1
        )
        self.assertSimilar(f_diff(interval.to_vector()), mps_cheb.to_vector())

    def test_gaussian_integral_1d(self):
        def equal_up_to_constant(arr1, arr2):
            difference = arr1 - arr2
            return np.allclose(difference, difference[0])

        start = -1
        stop = 2
        sites = 5
        order = 20
        f = lambda x: np.exp(-(x**2))
        f_intg = lambda x: (np.sqrt(np.pi) / 2) * erf(x)
        interval = RegularHalfOpenInterval(start, stop, 2**sites)
        mps_domain = mps_interval(interval)
        mps_cheb = chebyshev_approximation(
            f, order, mps_domain, differentiation_order=-1
        )
        assert equal_up_to_constant(f_intg(interval.to_vector()), mps_cheb.to_vector())

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
        start = -1
        stop = 2
        sites = 5
        order = 40
        interval = RegularHalfOpenInterval(start, stop, 2**sites)
        vector = interval.to_vector()
        func = lambda x: np.exp(-(x**2))
        t = 1e-15
        strategy = Strategy(tolerance=t, simplification_tolerance=t)
        mps_intv = mps_interval(interval)
        mps_domain = mps_tensor_sum([mps_intv, mps_intv])
        mps_cheb = chebyshev_approximation(func, order, mps_domain, strategy=strategy)
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


def position_mpo(a, b, n):
    left_tensor = np.zeros((1, 2, 2, 2))
    right_tensor = np.zeros((2, 2, 2, 1))
    middle_tensors = [np.zeros((2, 2, 2, 2)) for _ in range(n - 2)]
    dx = (b - a) / 2**n
    left_tensor[0, :, :, 0] = np.eye(2)
    left_tensor[0, :, :, 1] = np.array([[a, 0], [0, a + dx * 2 ** (n - 1)]])
    right_tensor[1, :, :, 0] = np.eye(2)
    right_tensor[0, :, :, 0] = np.array([[0, 0], [0, dx]])
    for i in range(len(middle_tensors)):
        middle_tensors[i][0, :, :, 0] = np.eye(2)
        middle_tensors[i][1, :, :, 1] = np.eye(2)
        middle_tensors[i][0, :, :, 1] = np.array([[0, 0], [0, dx * 2 ** (n - (i + 2))]])
    tensors = [left_tensor] + middle_tensors + [right_tensor]
    mpo = MPO(tensors)
    return mpo
