import numpy as np
import seemps
from seemps.state import CanonicalMPS, scprod, product_state
from seemps.expectation import (
    all_expectation1,
    expectation1,
    expectation2,
    product_expectation,
)
from ..tools import TestCase, run_over_random_uniform_mps, random_uniform_mps


def bit2state(b):
    if b:
        return [0, 1]
    else:
        return [1, 0]


class TestExpectation(TestCase):
    def test_scprod_is_antilinear_wrt_first_argument(self):
        a = -0.2 - 0.2j
        state1 = product_state([1, 0.3j], 3)
        state1_conj = product_state([1, -0.3j], 3)
        state2 = product_state([1.0, 1.0], 3)
        self.assertAlmostEqual(
            scprod(a * state1, state2), a.conjugate() * scprod(state1, state2)
        )
        self.assertAlmostEqual(scprod(state1, a * state2), a * scprod(state1, state2))
        self.assertAlmostEqual(scprod(state1, state2), scprod(state2, state1_conj))
        self.assertAlmostEqual(
            scprod(state1_conj, state2), scprod(state1, state2).conjugate()
        )

    def test_scprod_basis(self):
        #
        # Test that scprod() can be used to project onto basis states
        for nbits in range(1, 8):
            # We create a random MPS
            ψmps = seemps.state.random_uniform_mps(2, nbits, 2)
            ψwave = ψmps.to_vector()

            # We then create the basis of all states with well defined
            # values of the qubits
            conf = np.arange(0, 2**nbits, dtype=np.uint8)
            conf = np.reshape(conf, (2**nbits, 1))
            conf = np.unpackbits(conf, axis=1)

            # Finally, we loop over the basis states, verifying that the
            # scalar product is the projection onto the state
            for n, bits in enumerate(conf):
                proj = ψwave[n]
                ϕmps = seemps.state.product_state(map(bit2state, bits[-nbits:]))
                self.assertAlmostEqual(proj, scprod(ϕmps, ψmps))

    def test_norm_standard(self):
        #
        # Test the norm on our sample states
        for nbits in range(1, 8):
            self.assertAlmostEqual(
                seemps.state.GHZ(nbits).norm_squared(), 1.0, places=10
            )
            self.assertAlmostEqual(seemps.state.W(nbits).norm_squared(), 1.0, places=10)
            if nbits > 1:
                self.assertAlmostEqual(
                    seemps.state.AKLT(nbits).norm_squared(), 1.0, places=10
                )
                self.assertAlmostEqual(
                    seemps.state.factories.graph_state(nbits).norm_squared(),
                    1.0,
                    places=10,
                )

    def test_norm_random(self):
        # Test that the norm works on random states
        for nbits in range(1, 8):
            for _ in range(10):
                # We create a random MPS
                ψmps = seemps.state.random_uniform_mps(2, nbits, 2)
                ψwave = ψmps.to_vector()
                self.assertAlmostEqual(ψmps.norm_squared(), np.vdot(ψwave, ψwave))

    def test_expected1_standard(self):
        O = np.array([[0, 0], [0, 1]])
        for nbits in range(1, 8):
            ψGHZ = seemps.state.GHZ(nbits)
            ψW = seemps.state.W(nbits)
            for i in range(nbits):
                self.assertAlmostEqual(ψGHZ.expectation1(O, i), 0.5)
                self.assertAlmostEqual(ψW.expectation1(O, i), 1 / nbits)

    def test_expected1(self):
        O1 = np.array([[0.3, 1.0 + 0.2j], [1.0 - 0.2j, 0.5]])

        def expected1_ok(ϕ, canonical=False):
            if canonical:
                for i in range(ϕ.size):
                    expected1_ok(CanonicalMPS(ϕ, center=i), canonical=False)
            else:
                nrm2 = ϕ.norm_squared()
                for n in range(ϕ.size):
                    ψ = ϕ.copy()
                    ψ[n] = np.einsum("ij,kjl->kil", O1, ψ[n])
                    desired = np.vdot(ϕ.to_vector(), ψ.to_vector())
                    self.assertAlmostEqual(
                        desired / nrm2, expectation1(ϕ, O1, n) / nrm2
                    )
                    self.assertAlmostEqual(desired / nrm2, ϕ.expectation1(O1, n) / nrm2)

        run_over_random_uniform_mps(expected1_ok)
        run_over_random_uniform_mps(lambda ϕ: expected1_ok(ϕ, canonical=True))

    def test_expected1_density(self):
        def random_wavefunction(n):
            ψ = np.random.rand(n) - 0.5
            return ψ / np.linalg.norm(ψ)

        #
        # When we create a spin wave, 'O' detects the density of the
        # wave with the right magnitude
        O = np.array([[0, 0], [0, 1]])
        for nbits in range(2, 14):
            for _ in range(10):
                # We create a random MPS
                ψwave = random_wavefunction(nbits)
                ψmps = seemps.state.spin_wave(ψwave)
                ni = all_expectation1(ψmps, O)
                for i in range(nbits):
                    si = expectation1(ψmps, O, i)
                    self.assertAlmostEqual(si, ψwave[i] ** 2)
                    xi = ψmps.expectation1(O, i)
                    self.assertEqual(si, xi)
                    self.assertAlmostEqual(ni[i], si)

    def test_expected2_GHZ(self):
        σz = np.array([[1, 0], [0, -1]])
        for nbits in range(2, 8):
            ψGHZ = seemps.state.GHZ(nbits)
            for i in range(nbits - 1):
                self.assertAlmostEqual(expectation2(ψGHZ, σz, σz, i), 1)
                self.assertAlmostEqual(ψGHZ.expectation2(σz, σz, i), 1)

    def test_expected2(self):
        O1 = np.array([[0.3, 1.0 + 0.2j], [1.0 - 0.2j, 0.5]])
        O2 = np.array([[0.34, 0.4 - 0.7j], [0.4 + 0.7j, -0.6]])

        def expected2_ok(ϕ, canonical=False):
            if canonical:
                for i in range(ϕ.size):
                    CanonicalMPS(ϕ, center=i)
            nrm2 = ϕ.norm_squared()
            for n in range(1, ϕ.size):
                ψ = ϕ.copy()
                ψ[n - 1] = np.einsum("ij,kjl->kil", O1, ψ[n - 1])
                ψ[n] = np.einsum("ij,kjl->kil", O2, ψ[n])
                desired = scprod(ϕ, ψ)
                self.assertAlmostEqual(
                    desired / nrm2, expectation2(ϕ, O1, O2, n - 1) / nrm2
                )
                self.assertAlmostEqual(
                    desired / nrm2, ϕ.expectation2(O1, O2, n - 1) / nrm2
                )

        run_over_random_uniform_mps(expected2_ok)
        run_over_random_uniform_mps(lambda ϕ: expected2_ok(ϕ, canonical=True))

    def test_expectation2_with_same_site_is_product(self):
        state = random_uniform_mps(2, 10, rng=self.rng)
        σz = np.array([[1, 0], [0, -1]])
        σx = np.array([[0, 1], [1, 0]])
        self.assertAlmostEqual(state.expectation2(σz, σz, 1, 1), state.norm_squared())
        self.assertAlmostEqual(
            state.expectation2(σz, σx, 1, 1), state.expectation1(σz @ σx, 1)
        )

    def test_expectation2_sorts_site_indices(self):
        state = random_uniform_mps(2, 10, rng=self.rng)
        σz = np.array([[1, 0], [0, -1]])
        σx = np.array([[0, 1], [1, 0]])
        self.assertAlmostEqual(
            state.expectation2(σz, σx, 2, 1), state.expectation2(σx, σz, 1, 2)
        )

    def test_expectation2_over_separate_sites(self):
        state = random_uniform_mps(2, 3, rng=self.rng)
        σz = np.array([[1, 0], [0, -1]])
        σx = np.array([[0, 1], [1, 0]])
        v = state.to_vector()
        self.assertAlmostEqual(
            state.expectation2(σz, σx, 0, 2),
            np.vdot(v, np.kron(σz, np.kron(np.eye(2), σx)) @ v),
        )

    def test_product_expectation(self):
        state = random_uniform_mps(2, 3, rng=self.rng)
        σz = np.array([[1, 0], [0, -1]])
        σx = np.array([[0, 1], [1, 0]])
        v = state.to_vector()
        exact_value = np.vdot(v, np.kron(σz, np.kron(σz, σz @ σx)) @ v)
        self.assertAlmostEqual(
            product_expectation(state, [σz, σx, σz @ σx]), exact_value
        )

    def test_product_expectation_checks_sizes(self):
        state = random_uniform_mps(2, 3, rng=self.rng)
        σz = np.array([[1, 0], [0, -1]])
        with self.assertRaises(Exception):
            product_expectation(state, [σz] * 4)
