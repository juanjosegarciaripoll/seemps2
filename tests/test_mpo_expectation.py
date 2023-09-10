from seemps.hamiltonians import ConstantTIHamiltonian
from seemps import MPO, σx, σy
from .tools import *


class TestMPOExpectation(TestCase):
    def test_mpo_expected_local_one_site(self):
        """Ensure expectation of a single local operator works."""
        H = MPO([σx.reshape(1, 2, 2, 1)])
        psi = random_mps(2, 1, rng=self.rng)
        O = σx
        v = psi.to_vector()
        self.assertAlmostEqual(H.expectation(psi), np.vdot(v, O @ v))

    def test_mpo_expected_operator_order(self):
        """Ensure expectation of a two different local operators are done in order."""
        H = MPO([σx.reshape(1, 2, 2, 1), σy.reshape(1, 2, 2, 1)])
        psi = random_mps(2, 2, rng=self.rng)
        O = np.kron(σx, σy)
        v = psi.to_vector()
        self.assertAlmostEqual(H.expectation(psi), np.vdot(v, O @ v))

    def test_mpo_expected_rejects_mps_sum(self):
        H = MPO([σx.reshape(1, 2, 2, 1)] * 2)
        psi = random_mps(2, 1, rng=self.rng)
        with self.assertRaises(Exception):
            H.expectation(psi + psi)
