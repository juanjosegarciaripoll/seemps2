import numpy as np
import scipy.sparse.linalg  # type: ignore
from seemps.optimization.dmrg import QuadraticForm, dmrg
from seemps.hamiltonians import ConstantTIHamiltonian, HeisenbergHamiltonian
from seemps.cython import _contract_last_and_first
from seemps.state import random_uniform_mps, product_state, CanonicalMPS
from seemps.operators import MPO
from seemps.typing import DenseOperator
from ..tools import TestCase


class TestQuadraticForm(TestCase):
    Sz: DenseOperator = np.diag([1, -1])
    Sx: DenseOperator = np.array([[0, 1], [1, 0]])

    def setUp(self) -> None:
        return super().setUp()

    def test_quadratic_form_checks_mpo_size(self):
        mpo = MPO([np.ones((1, 2, 2, 1))] * 3)
        mps = random_uniform_mps(2, 4, rng=self.rng)
        with self.assertRaises(Exception):
            QuadraticForm(mpo, mps)  # type: ignore

    def test_quadratic_form_checks_mpo_dimensions(self):
        mpo = MPO([np.ones((1, 2, 2, 1))] * 3)
        mps = random_uniform_mps(3, 3, rng=self.rng)
        with self.assertRaises(Exception):
            QuadraticForm(mpo, mps)  # type: ignore

    def test_quadratic_form_two_sites(self):
        H = ConstantTIHamiltonian(size=2, interaction=np.kron(self.Sz, self.Sx))
        Hmpo = H.to_mpo()
        state = CanonicalMPS(random_uniform_mps(2, 2, D=2, rng=self.rng), center=0)

        Q = QuadraticForm(Hmpo, state, start=0)
        Hop = Q.two_site_Hamiltonian(0)
        AB = _contract_last_and_first(state[0], state[1])
        HopAB = Hop @ AB.reshape(-1)

        self.assertEqual(HopAB.shape, (AB.size,))

        expected = np.vdot(AB, HopAB)
        exact_expected = Hmpo.expectation(state)
        self.assertAlmostEqual(expected, exact_expected)  # type: ignore

        HAB = H.to_matrix() @ AB.reshape(-1)
        self.assertSimilar(HAB, HopAB)

    def test_quadratic_form_three_sites_start_zero(self):
        H = ConstantTIHamiltonian(size=3, interaction=np.kron(self.Sz, self.Sx))
        Hmpo = H.to_mpo()
        state = CanonicalMPS(random_uniform_mps(2, 3, D=2, rng=self.rng), center=0)

        Q = QuadraticForm(Hmpo, state, start=0)
        Hop = Q.two_site_Hamiltonian(0)
        AB = _contract_last_and_first(state[0], state[1])
        HopAB = Hop @ AB.reshape(-1)

        self.assertEqual(HopAB.shape, (AB.size,))

        expected = np.vdot(AB, HopAB)
        exact_expected = Hmpo.expectation(state)
        self.assertAlmostEqual(expected, exact_expected)  # type: ignore

    def test_quadratic_form_three_sites_start_one(self):
        H = ConstantTIHamiltonian(size=3, interaction=np.kron(self.Sz, self.Sx))
        Hmpo = H.to_mpo()
        state = CanonicalMPS(random_uniform_mps(2, 3, D=2, rng=self.rng), center=0)

        Q = QuadraticForm(Hmpo, state, start=1)
        Hop = Q.two_site_Hamiltonian(1)
        AB = _contract_last_and_first(state[1], state[2])
        HopAB = Hop @ AB.reshape(-1)

        self.assertEqual(HopAB.shape, (AB.size,))

        expected = np.vdot(AB, HopAB)
        exact_expected = Hmpo.expectation(state)
        self.assertAlmostEqual(expected, exact_expected)  # type: ignore


class TestDMRG(TestCase):
    Sz: DenseOperator = np.diag([1.0, -1.0])
    Sx: DenseOperator = np.array([[0.0, 1.0], [1.0, 0.0]])

    def test_dmrg_on_Ising_two_sites(self):
        """Check we can compute ground state of Sz * Sz on two sites"""
        aux = np.kron(self.Sz, self.Sz)
        assert aux is not None
        H = ConstantTIHamiltonian(size=2, interaction=-np.kron(self.Sz, self.Sz))
        Hmpo = H.to_mpo()
        result = dmrg(Hmpo, guess=self.random_uniform_mps(2, 2))
        self.assertAlmostEqual(result.energy, -1)
        v = result.state.to_vector()
        self.assertAlmostEqual(v[0] ** 2 + v[3] ** 2, 1.0)
        self.assertAlmostEqual(v[1] ** 2 + v[2] ** 2, 0.0)

    def test_dmrg_on_Ising_three_sites(self):
        """Check we can compute ground state of Sz * Sz on two sites"""
        H = ConstantTIHamiltonian(size=3, interaction=-np.kron(self.Sz, self.Sz))
        Hmpo = H.to_mpo()
        result = dmrg(Hmpo, guess=self.random_uniform_mps(2, 3))
        self.assertAlmostEqual(result.energy, -2)
        self.assertAlmostEqual(Hmpo.expectation(result.state), -2)

    def test_dmrg_on_Heisenberg_five_sites(self):
        """Check we can compute ground state of Sz * Sz on two sites"""
        H = HeisenbergHamiltonian(size=5, field=[0.0, 0.0, 0.1])
        Hmpo = H.to_mpo()
        result = dmrg(Hmpo, guess=self.random_uniform_mps(2, 5))
        E, exact_v = scipy.sparse.linalg.eigsh(H.to_matrix(), k=1, which="SA")
        self.assertAlmostEqual(result.energy, E[0])
        v = result.state.to_vector()
        self.assertAlmostEqual(abs(np.vdot(v, exact_v)), np.float64(1.0))

    def test_dmrg_works_with_hamiltonian(self):
        """Check we can compute ground state of Sz * Sz on two sites"""
        H = ConstantTIHamiltonian(size=3, interaction=-np.kron(self.Sz, self.Sz))
        Hmpo = H.to_mpo()
        result = dmrg(H, guess=self.random_uniform_mps(2, 3))
        self.assertAlmostEqual(result.energy, -2)
        self.assertAlmostEqual(Hmpo.expectation(result.state), -2)

    def test_dmrg_uses_guess_with_canonical_centers_other_than_zero(self):
        """Check we can compute ground state of Sz * Sz on two sites"""
        H = ConstantTIHamiltonian(size=2, interaction=-np.kron(self.Sz, self.Sz))
        mps = CanonicalMPS(product_state([1.0, 1.0], 2), center=-1, normalize=True)
        Hmpo = H.to_mpo()
        result = dmrg(Hmpo, guess=mps, maxiter=100)
        self.assertAlmostEqual(result.energy, -1)
        v = result.state.to_vector()
        self.assertAlmostEqual(v[0] ** 2 + v[3] ** 2, 1.0)
        self.assertAlmostEqual(v[1] ** 2 + v[2] ** 2, 0.0)
