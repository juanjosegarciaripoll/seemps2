import numpy as np
from numpy.polynomial.polynomial import Polynomial
from seemps.state import MPS, DEFAULT_STRATEGY, NO_TRUNCATION, CanonicalMPS
from seemps.analysis.polynomials import _mps_x_tensor, mps_from_polynomial
from seemps.analysis.mesh import RegularInterval, Interval
from ..tools import TestCase


class TestMonomialsCollection(TestCase):
    def assertContainsMonomials(self, L: int, mps: MPS, domain: Interval, first: bool):
        x = domain.to_vector()
        for m in range(L):
            xm_mps: MPS = mps.copy()
            if first:
                xm_mps[0] = xm_mps[0][[m], :, :]
            else:
                xm_mps[-1] = xm_mps[-1][:, :, [m]]
            if np.all(np.isclose(xm_mps.to_vector(), x**m)):
                continue
            raise AssertionError(f"MPS fails to reproduce monomial of order {m}")

    def test_all_monomials_up_to_fourth_order_from_end(self):
        N = 5  # qubits
        L = 5  # one plust the last order
        domain = RegularInterval(0, 1, 2**N, endpoint_right=True)
        xL_mps = _mps_x_tensor(L, domain, first=False)
        self.assertContainsMonomials(L, xL_mps, domain, first=False)

    def test_all_monomials_up_to_fourth_order_from_start(self):
        N = 5  # qubits
        L = 5  # one plust the last order
        domain = RegularInterval(0, 1, 2**N, endpoint_right=True)
        xL_mps = _mps_x_tensor(L, domain, first=True)
        self.assertContainsMonomials(L, xL_mps, domain, first=True)


class TestPolynomialFunction(TestCase):
    N = 5
    domain = RegularInterval(0, 1, 2**5, endpoint_right=True)

    def assertSimilarPolynomial(self, p: Polynomial, p_mps: MPS):
        x = self.domain.to_vector()
        self.assertSimilar(p(x), p_mps.to_vector())

    def test_polynomial_output_type(self):
        p = Polynomial([1])
        mps1 = mps_from_polynomial(p, self.domain, strategy=DEFAULT_STRATEGY)
        self.assertIsInstance(mps1, CanonicalMPS)
        mps2 = mps_from_polynomial(p, self.domain, strategy=NO_TRUNCATION)
        self.assertNotIsInstance(mps2, CanonicalMPS)
        self.assertIsInstance(mps2, MPS)

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

    def test_constant_polynomial_mps_first_true(self):
        p = Polynomial([1])
        p_mps = mps_from_polynomial(p, self.domain, first=True)
        self.assertSimilarPolynomial(p, p_mps)

    def test_first_order_polynomial_mps_first_true(self):
        p = Polynomial([1, -3])
        p_mps = mps_from_polynomial(p, self.domain, first=True)
        self.assertSimilarPolynomial(p, p_mps)

    def test_third_order_polynomial_mps_first_true(self):
        p = Polynomial([1, -2, 0.4, -0.25])
        p_mps = mps_from_polynomial(p, self.domain, first=True)
        self.assertSimilarPolynomial(p, p_mps)
