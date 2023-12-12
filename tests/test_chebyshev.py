import numpy as np
from scipy.special import erf
from seemps.analysis import (
    RegularHalfOpenInterval,
    chebyshev_approximation_clenshaw,
    chebyshev_coefficients,
    differentiate_chebyshev_coefficients,
    integrate_chebyshev_coefficients,
    mps_tensor_sum,
    mps_tensor_product,
    mps_interval,
)

from .tools import TestCase


class TestChebyshev(TestCase):
    def test_chebyshev_coefficients(self):
        f = lambda x: np.exp(x)
        cheb_coeffs = chebyshev_coefficients(f, -1, 1, 15)
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
        start = -2
        stop = 2
        sites = 5
        order = 20
        f = lambda x: np.exp(-(x**2))
        coefficients = chebyshev_coefficients(f, start, stop, order)
        interval = RegularHalfOpenInterval(start, stop, 2**sites)
        mps = chebyshev_approximation_clenshaw(
            coefficients, mps_interval(interval, rescale=True)
        )
        self.assertSimilar(f(interval.to_vector()), mps.to_vector())

    def test_gaussian_1d_derivative(self):
        start = -1
        stop = 3
        sites = 5
        order = 25
        f = lambda x: np.exp(-(x**2))
        f_diff = lambda x: -2 * x * np.exp(-(x**2))
        coefficients = differentiate_chebyshev_coefficients(
            chebyshev_coefficients(f, start, stop, order), start, stop
        )
        interval = RegularHalfOpenInterval(start, stop, 2**sites)
        mps = chebyshev_approximation_clenshaw(
            coefficients, mps_interval(interval, rescale=True)
        )
        self.assertSimilar(f_diff(interval.to_vector()), mps.to_vector())

    def test_gaussian_1d_integral(self):
        def equal_up_to_constant(arr1, arr2):
            difference = arr1 - arr2
            return np.allclose(difference, difference[0])

        start = -2
        stop = 2
        sites = 5
        order = 20
        f = lambda x: np.exp(-(x**2))
        f_intg = lambda x: np.sqrt(np.pi) / 2 * erf(x)
        coefficients = integrate_chebyshev_coefficients(
            chebyshev_coefficients(f, start, stop, order), start, stop
        )
        interval = RegularHalfOpenInterval(start, stop, 2**sites)
        mps = chebyshev_approximation_clenshaw(
            coefficients, mps_interval(interval, rescale=True)
        )
        assert equal_up_to_constant(f_intg(interval.to_vector()), mps.to_vector())

    def test_tensor_product(self):
        start = -2
        stop = 2
        sites = 5
        interval = RegularHalfOpenInterval(start, stop, 2**sites)
        vector = interval.to_vector()
        mps = mps_interval(interval)
        mps_tensor = mps_tensor_product([mps, mps])
        Z_mps = mps_tensor.to_vector().reshape((2**sites, 2**sites))
        X, Y = np.meshgrid(vector, vector)
        self.assertSimilar(Z_mps, X * Y)

    def test_tensor_sum(self):
        start = -2
        stop = 2
        sites = 5
        interval = RegularHalfOpenInterval(start, stop, 2**sites)
        vector = interval.to_vector()
        mps = mps_interval(interval)
        mps_tensor = mps_tensor_sum([mps, mps])
        Z_mps = mps_tensor.to_vector().reshape((2**sites, 2**sites))
        X, Y = np.meshgrid(vector, vector)
        self.assertSimilar(Z_mps, X + Y)
