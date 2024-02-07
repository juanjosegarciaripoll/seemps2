import numpy as np
from numpy.polynomial.polynomial import Polynomial
from seemps.state import MPS
from seemps.analysis.polynomials import _mps_x_tensor, mps_from_polynomial
from seemps.analysis import RegularClosedInterval, Interval
from ..tools import TestCase


class TestMonomialsCollection(TestCase):
    def assertContainsMonomials(self, L: int, mps: MPS, domain: Interval):
        x = domain.to_vector()
        for m in range(L):
            xm_mps: MPS = mps.copy()
            xm_mps[-1] = xm_mps[-1][:, :, [m]]
            if np.all(np.isclose(xm_mps.to_vector(), x**m)):
                continue
            raise AssertionError(f"MPS fails to reproduce monomial of order {m}")

    def test_all_monomials_up_to_fourth_order(self):
        N = 5  # qubits
        L = 5  # one plust the last order
        domain = RegularClosedInterval(0, 1, 2**N)
        xL_mps = _mps_x_tensor(L, domain)
        self.assertContainsMonomials(L, xL_mps, domain)


class TestPolynomialFunction(TestCase):
    N = 5
    domain = RegularClosedInterval(0, 1, 2**5)

    def assertSimilarPolynomial(self, p: Polynomial, p_mps: MPS):
        x = self.domain.to_vector()
        self.assertSimilar(p(x), p_mps.to_vector())

    def test_constant_polynomial_mps(self):
        p = Polynomial([1])
        p_mps = mps_from_polynomial(p, self.domain)
        self.assertSimilarPolynomial(p, p_mps)

    def test_first_order_polynomial_mps(self):
        p = Polynomial([1, -3])
        p_mps = mps_from_polynomial(p, self.domain)
        self.assertSimilarPolynomial(p, p_mps)

    def test_third_order_polynomial_mps(self):
        p = Polynomial([1, -2, 0.4, -0.25])
        p_mps = mps_from_polynomial(p, self.domain)
        self.assertSimilarPolynomial(p, p_mps)
