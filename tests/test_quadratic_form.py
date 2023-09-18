import numpy as np
from seemps.optimization.dmrg import QuadraticForm, dmrg
from seemps.hamiltonians import ConstantTIHamiltonian, HeisenbergHamiltonian
from seemps.state._contractions import _contract_last_and_first
from seemps.state import random_uniform_mps
from .tools import *
from seemps.tools import DEBUG
import scipy.sparse.linalg


class TestQuadraticForm(TestCase):
    Sz = np.diag([1, -1])
    Sx = np.array([[0, 1], [1, 0]])

    def setUp(self) -> None:
        return super().setUp()

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
        self.assertAlmostEqual(expected, exact_expected)

        HAB = H.tomatrix() @ AB.reshape(-1)
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
        self.assertAlmostEqual(expected, exact_expected)

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
        self.assertAlmostEqual(expected, exact_expected)


class TestDMRG(TestCase):
    Sz = np.diag([1, -1])
    Sx = np.array([[0, 1], [1, 0]])

    def test_dmrg_on_Ising_two_sites(self):
        """Check we can compute ground state of Sz * Sz on two sites"""
        H = ConstantTIHamiltonian(size=2, interaction=-np.kron(self.Sz, self.Sz))
        Hmpo = H.to_mpo()
        result = dmrg(Hmpo)
        self.assertAlmostEqual(result.energy, -1)
        v = result.state.to_vector()
        self.assertAlmostEqual(v[0] ** 2 + v[3] ** 2, 1.0)
        self.assertAlmostEqual(v[1] ** 2 + v[2] ** 2, 1.0)

    def test_dmrg_on_Ising_two_sites(self):
        """Check we can compute ground state of Sz * Sz on two sites"""
        H = ConstantTIHamiltonian(size=3, interaction=-np.kron(self.Sz, self.Sz))
        Hmpo = H.to_mpo()
        result = dmrg(Hmpo)
        self.assertAlmostEqual(result.energy, -2)
        self.assertAlmostEqual(Hmpo.expectation(result.state), -2)

    def test_dmrg_on_Heisenberg_five_sites(self):
        """Check we can compute ground state of Sz * Sz on two sites"""
        H = HeisenbergHamiltonian(size=5, field=[0.0, 0.0, 0.1])
        Hmpo = H.to_mpo()
        result = dmrg(Hmpo)
        E, exact_v = scipy.sparse.linalg.eigsh(H.tomatrix(), k=1, which="SA")
        self.assertAlmostEqual(result.energy, E[0])
        v = result.state.to_vector()
        print(np.linalg.norm(v), np.linalg.norm(exact_v))
        self.assertAlmostEqual(abs(np.vdot(v, exact_v)), 1.0)
