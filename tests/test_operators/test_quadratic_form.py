import numpy as np
from seemps.optimization.dmrg import QuadraticForm
from seemps.hamiltonians import ConstantTIHamiltonian
from seemps.cython import _contract_last_and_first
from seemps.state import CanonicalMPS
from seemps.operators import MPO
from seemps.typing import DenseOperator
from ..tools import SeeMPSTestCase


class TestQuadraticForm(SeeMPSTestCase):
    Sz: DenseOperator = np.diag([1, -1])
    Sx: DenseOperator = np.array([[0, 1], [1, 0]])

    def setUp(self) -> None:
        return super().setUp()

    def _make_H(self, size: int) -> ConstantTIHamiltonian:
        return ConstantTIHamiltonian(size=size, interaction=np.kron(self.Sz, self.Sx))

    def _canonical(self, size: int, D: int = 2, center: int = 0) -> CanonicalMPS:
        return CanonicalMPS(self.random_uniform_mps(2, size, D=D), center=center)

    def test_quadratic_form_checks_mpo_size(self):
        mpo = MPO([np.ones((1, 2, 2, 1))] * 3)
        mps = self.random_uniform_mps(2, 4)
        with self.assertRaises(Exception):
            QuadraticForm(mpo, mps)  # type: ignore

    def test_quadratic_form_checks_mpo_dimensions(self):
        mpo = MPO([np.ones((1, 2, 2, 1))] * 3)
        mps = self.random_uniform_mps(3, 3)
        with self.assertRaises(Exception):
            QuadraticForm(mpo, mps)  # type: ignore

    def test_quadratic_form_two_sites(self):
        H = self._make_H(2)
        Hmpo = H.to_mpo()
        state = CanonicalMPS(self.random_uniform_mps(2, 2, D=2), center=0)
        Q = QuadraticForm(Hmpo, state, start=0)
        Hop = Q.two_site_operator(0)
        AB = _contract_last_and_first(state[0], state[1])
        HopAB = Hop @ AB.reshape(-1)
        self.assertEqual(HopAB.shape, (AB.size,))
        self.assertAlmostEqual(np.vdot(AB, HopAB), Hmpo.expectation(state))  # type: ignore
        self.assertSimilar(H.to_matrix() @ AB.reshape(-1), HopAB)

    def test_quadratic_form_three_sites_start_zero(self):
        H = self._make_H(3)
        Hmpo = H.to_mpo()
        state = CanonicalMPS(self.random_uniform_mps(2, 3, D=2), center=0)
        Q = QuadraticForm(Hmpo, state, start=0)
        AB = _contract_last_and_first(state[0], state[1])
        HopAB = Q.two_site_operator(0) @ AB.reshape(-1)
        self.assertAlmostEqual(np.vdot(AB, HopAB), Hmpo.expectation(state))  # type: ignore

    def test_quadratic_form_three_sites_start_one(self):
        H = self._make_H(3)
        Hmpo = H.to_mpo()
        state = CanonicalMPS(self.random_uniform_mps(2, 3, D=2), center=0)
        Q = QuadraticForm(Hmpo, state, start=1)
        AB = _contract_last_and_first(state[1], state[2])
        HopAB = Q.two_site_operator(1) @ AB.reshape(-1)
        self.assertAlmostEqual(np.vdot(AB, HopAB), Hmpo.expectation(state))  # type: ignore
