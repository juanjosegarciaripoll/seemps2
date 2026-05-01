import numpy as np
import scipy.sparse.linalg  # type: ignore
from seemps.optimization.dmrg import dmrg
from seemps.hamiltonians import ConstantTIHamiltonian, HeisenbergHamiltonian
from seemps.state import product_state, CanonicalMPS
from seemps.typing import DenseOperator
from ..tools import SeeMPSTestCase


class TestDMRG(SeeMPSTestCase):
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
        """Check we can compute ground state of Sz * Sz on three sites"""
        H = ConstantTIHamiltonian(size=5, interaction=-np.kron(self.Sz, self.Sz))
        mps = CanonicalMPS(product_state([1.0, 1.0], 5), center=3, normalize=True)
        Hmpo = H.to_mpo()
        result = dmrg(Hmpo, guess=mps, maxiter=100)
        self.assertAlmostEqual(result.energy, -4)
        v = result.state.to_vector()
        self.assertAlmostEqual(v[0] ** 2 + v[-1] ** 2, 1.0)
        self.assertAlmostEqual(sum(v[1:-1] ** 2), 0.0)
