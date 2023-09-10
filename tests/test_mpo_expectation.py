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

    def test_mpo_expected_with_left_orthogonal_state(self):
        H = MPO([σx.reshape(1, 2, 2, 1)] * 10)
        state = CanonicalMPS(random_mps(2, 10, truncate=True, rng=self.rng), center=0)
        O = H.tomatrix()
        v = state.to_vector()
        self.assertSimilar(H.expectation(state), np.vdot(v, O @ v))

    def test_mpo_expected_with_right_orthogonal_state(self):
        H = MPO([σx.reshape(1, 2, 2, 1)] * 10)
        state = CanonicalMPS(random_mps(2, 10, truncate=True, rng=self.rng), center=9)
        O = H.tomatrix()
        v = state.to_vector()
        self.assertSimilar(H.expectation(state), np.vdot(v, O @ v))

    def test_mpo_expected_with_middle_orthogonal_state(self):
        H = MPO([σx.reshape(1, 2, 2, 1)] * 10)
        state = CanonicalMPS(random_mps(2, 10, truncate=True, rng=self.rng), center=4)
        O = H.tomatrix()
        v = state.to_vector()
        self.assertSimilar(H.expectation(state), np.vdot(v, O @ v))
