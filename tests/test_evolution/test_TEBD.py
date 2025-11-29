import numpy as np
import scipy
from seemps.state import MPS, CanonicalMPS, DEFAULT_STRATEGY, NO_TRUNCATION
from seemps.evolution.trotter import (
    PairwiseUnitaries,
    Trotter,
    Trotter2ndOrder,
    Trotter3rdOrder,
)
from seemps.hamiltonians import HeisenbergHamiltonian
from .problem import EvolutionTestCase


def random_wavefunction(n):
    ψ = np.random.rand(n) - 0.5
    return ψ / np.linalg.norm(ψ)


class TestPairwiseUnitaries(EvolutionTestCase):
    def test_pairwise_unitaries_matrices(self):
        """Check that the nearest-neighbor unitary matrices are built properly."""
        dt = 0.33
        H = HeisenbergHamiltonian(4)
        pairwiseU = PairwiseUnitaries(H, dt, DEFAULT_STRATEGY)
        exactU = scipy.linalg.expm(-1j * dt * self.Heisenberg2)
        self.assertSimilar(pairwiseU.U[0], exactU)
        self.assertSimilar(pairwiseU.U[1], exactU)
        self.assertSimilar(pairwiseU.U[2], exactU)

    def test_pairwise_unitaries_two_sites(self):
        """Verify the exact action of the PairwiseUnitaries on two sites."""
        dt = 0.33
        H = HeisenbergHamiltonian(2)
        exactU = scipy.linalg.expm(-1j * dt * self.Heisenberg2)
        pairwiseU = PairwiseUnitaries(H, dt, DEFAULT_STRATEGY)
        mps = self.random_initial_state(2)
        self.assertSimilar(
            pairwiseU.U[0].reshape(4, 4) @ mps.to_vector(), exactU @ mps.to_vector()
        )
        self.assertSimilar(pairwiseU.apply(mps).to_vector(), exactU @ mps.to_vector())

    def test_pairwise_unitaries_three_sites(self):
        """Verify the exact action of the PairwiseUnitaries on three sites."""
        dt = 0.33
        H = HeisenbergHamiltonian(3)
        exactU12 = np.kron(scipy.linalg.expm(-1j * dt * self.Heisenberg2), np.eye(2))
        exactU23 = np.kron(np.eye(2), scipy.linalg.expm(-1j * dt * self.Heisenberg2))
        pairwiseU = PairwiseUnitaries(H, dt, DEFAULT_STRATEGY)
        mps = self.random_initial_state(3)
        #
        # When center = 0, unitaries are applied left to right
        self.assertSimilar(
            pairwiseU.apply(CanonicalMPS(mps, center=0)).to_vector(),
            exactU23 @ exactU12 @ mps.to_vector(),
        )
        #
        # Otherwise, they are applied right to left
        self.assertSimilar(
            pairwiseU.apply(CanonicalMPS(mps, center=2)).to_vector(),
            exactU12 @ exactU23 @ mps.to_vector(),
        )

    def test_pairwise_unitaries_four_sites(self):
        """Verify the exact action of the PairwiseUnitaries on four sites."""
        dt = 0.33
        H = HeisenbergHamiltonian(4)
        exactU12 = np.kron(scipy.linalg.expm(-1j * dt * self.Heisenberg2), np.eye(4))
        exactU23 = np.kron(
            np.eye(2),
            np.kron(scipy.linalg.expm(-1j * dt * self.Heisenberg2), np.eye(2)),
        )
        exactU34 = np.kron(np.eye(4), scipy.linalg.expm(-1j * dt * self.Heisenberg2))
        pairwiseU = PairwiseUnitaries(H, dt, NO_TRUNCATION)
        mps = self.random_initial_state(4)
        #
        # When center = 0, unitaries are applied left to right
        self.assertSimilar(
            pairwiseU.apply(CanonicalMPS(mps, center=0)).to_vector(),
            exactU34 @ exactU23 @ exactU12 @ mps.to_vector(),
        )
        #
        # Otherwise, they are applied right to left
        self.assertSimilar(
            pairwiseU.apply(CanonicalMPS(mps, center=2)).to_vector(),
            exactU12 @ exactU23 @ exactU34 @ mps.to_vector(),
        )

    def test_pairwise_unitaries_center_strategy(self):
        """Verify how PairwiseUnitaries decides the order of application."""
        dt = 0.33
        N = 7
        H = HeisenbergHamiltonian(N)
        pairwiseU = PairwiseUnitaries(H, dt, NO_TRUNCATION)
        mps = self.random_initial_state(N)
        mps_from_left = pairwiseU.apply(CanonicalMPS(mps, center=0))
        mps_from_right = pairwiseU.apply(CanonicalMPS(mps, center=N - 1))
        self.assertSimilar(mps_from_left, pairwiseU.apply(CanonicalMPS(mps, center=0)))
        self.assertSimilar(mps_from_left, pairwiseU.apply(CanonicalMPS(mps, center=1)))
        self.assertSimilar(mps_from_left, pairwiseU.apply(CanonicalMPS(mps, center=2)))
        self.assertSimilar(mps_from_right, pairwiseU.apply(CanonicalMPS(mps, center=3)))
        self.assertSimilar(mps_from_right, pairwiseU.apply(CanonicalMPS(mps, center=4)))
        self.assertSimilar(mps_from_right, pairwiseU.apply(CanonicalMPS(mps, center=5)))
        self.assertSimilar(mps_from_right, pairwiseU.apply(CanonicalMPS(mps, center=6)))


class TestTrotter(EvolutionTestCase):
    def test_trotter_abstract_methods_signal_error(self):
        with self.assertRaises(Exception):
            U = Trotter()  # type: ignore
        from unittest.mock import patch

        p = patch.multiple(Trotter, __abstractmethods__=set())
        p.start()
        mps = self.random_initial_state(2)
        U = Trotter()  # type: ignore
        p.stop()
        with self.assertRaises(Exception):
            U.apply(mps)
        with self.assertRaises(Exception):
            U.apply_inplace(mps)
        with self.assertRaises(Exception):
            U @ mps  # type: ignore


class TestTrotter2nd(EvolutionTestCase):
    def test_trotter_2nd_order_two_sites(self):
        dt = 0.33
        trotterU = Trotter2ndOrder(HeisenbergHamiltonian(2), dt)
        U12 = scipy.linalg.expm(-1j * dt * self.Heisenberg2)
        mps = self.random_initial_state(2)
        self.assertSimilar(trotterU.apply(mps).to_vector(), U12 @ mps.to_vector())

    def test_trotter_2nd_matmul_is_equivalent_to_apply(self):
        trotterU = Trotter2ndOrder(HeisenbergHamiltonian(2), 0.33)
        mps = self.random_initial_state(2)
        self.assertSimilar(trotterU.apply(mps), trotterU @ mps)

    def test_trotter_2nd_apply_in_place_tries_to_reuse_mps(self):
        trotterU = Trotter2ndOrder(HeisenbergHamiltonian(2), 0.33)
        a = CanonicalMPS(self.random_initial_state(2))
        b = MPS(a.copy())
        self.assertTrue(trotterU.apply_inplace(a) is a)
        self.assertTrue(trotterU.apply(b) is not b)

    def test_trotter_2nd_order_three_sites(self):
        dt = 0.33
        trotterU = Trotter2ndOrder(HeisenbergHamiltonian(3), dt)
        U2 = scipy.linalg.expm(-0.5j * dt * self.Heisenberg2)
        U23 = np.kron(np.eye(2), U2)
        U12 = np.kron(U2, np.eye(2))
        mps = self.random_initial_state(3)
        self.assertSimilar(
            trotterU.apply(mps).to_vector(),
            U12 @ (U23 @ (U23 @ (U12 @ mps.to_vector()))),
        )

    def test_trotter_2nd_order_four_sites(self):
        dt = 0.33
        trotterU = Trotter2ndOrder(HeisenbergHamiltonian(4), dt)
        U2 = scipy.linalg.expm(-0.5j * dt * self.Heisenberg2)
        U34 = np.kron(np.eye(4), U2)
        U23 = np.kron(np.eye(2), np.kron(U2, np.eye(2)))
        U12 = np.kron(U2, np.eye(4))
        mps = self.random_initial_state(4)
        self.assertSimilar(
            trotterU.apply(mps).to_vector(),
            U12 @ (U23 @ (U34 @ (U34 @ (U23 @ (U12 @ mps.to_vector()))))),
        )


class TestTrotter3rd(EvolutionTestCase):
    def test_trotter_3rd_order_two_sites(self):
        dt = 0.33
        trotterU = Trotter3rdOrder(HeisenbergHamiltonian(2), dt)
        U12 = scipy.linalg.expm(-1j * dt * self.Heisenberg2)
        mps = self.random_initial_state(2)
        self.assertSimilar(trotterU.apply(mps).to_vector(), U12 @ mps.to_vector())

    def test_trotter_3rd_matmul_is_equivalent_to_apply(self):
        trotterU = Trotter3rdOrder(HeisenbergHamiltonian(2), 0.33)
        mps = self.random_initial_state(2)
        self.assertSimilar(trotterU.apply(mps), trotterU @ mps)

    def test_trotter_3rd_apply_in_place_tries_to_reuse_mps(self):
        trotterU = Trotter3rdOrder(HeisenbergHamiltonian(2), 0.33)
        a = CanonicalMPS(self.random_initial_state(2))
        b = MPS(a.copy())
        self.assertTrue(trotterU.apply_inplace(a) is a)
        self.assertTrue(trotterU.apply(b) is not b)

    def test_trotter_3rd_order_three_sites(self):
        dt = 0.33
        trotterU = Trotter3rdOrder(HeisenbergHamiltonian(3), dt)
        U2half = scipy.linalg.expm(-0.5j * dt * self.Heisenberg2)
        U2 = scipy.linalg.expm(-0.25j * dt * self.Heisenberg2)
        U23 = np.kron(np.eye(2), U2)
        U12 = np.kron(U2, np.eye(2))
        U23half = np.kron(np.eye(2), U2half)
        U12half = np.kron(U2half, np.eye(2))
        mps = self.random_initial_state(3)
        self.assertSimilar(
            trotterU.apply(mps).to_vector(),
            U23 @ (U12 @ (U12half @ (U23half @ (U23 @ (U12 @ mps.to_vector()))))),
        )
