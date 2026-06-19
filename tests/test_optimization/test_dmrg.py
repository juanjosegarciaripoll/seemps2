import numpy as np
import scipy.sparse.linalg  # type: ignore
from seemps.optimization.dmrg import dmrg, _convergence_reason
from seemps.optimization.descent import OptimizeResults
from seemps.hamiltonians import ConstantTIHamiltonian, HeisenbergHamiltonian
from seemps.operators import MPO
from seemps.state import MPS, product_state, CanonicalMPS, DEFAULT_STRATEGY
from seemps.typing import DenseOperator
from .tools import TestOptimizeCase
from ..tools import SeeMPSTestCase


class TestDMRG(TestOptimizeCase):
    Sz: DenseOperator = np.diag([1.0, -1.0])
    Sx: DenseOperator = np.array([[0.0, 1.0], [1.0, 0.0]])

    def solve(self, H: MPO, state: MPS, **kwdargs) -> OptimizeResults:
        return dmrg(H, guess=state, **kwdargs)

    # Correctness against exact ground states
    def test_dmrg_on_Ising_two_sites(self):
        """Check we can compute ground state of Sz * Sz on two sites"""
        H = ConstantTIHamiltonian(size=2, interaction=-np.kron(self.Sz, self.Sz))
        Hmpo = H.to_mpo()
        result = dmrg(Hmpo, guess=self.random_uniform_mps(2, 2))
        self.assertAlmostEqual(result.energy, -1)
        v = result.state.to_vector()
        self.assertAlmostEqual(v[0] ** 2 + v[3] ** 2, 1.0)
        self.assertAlmostEqual(v[1] ** 2 + v[2] ** 2, 0.0)

    def test_dmrg_on_Ising_three_sites(self):
        """Check we can compute ground state of Sz * Sz on three sites"""
        H = ConstantTIHamiltonian(size=3, interaction=-np.kron(self.Sz, self.Sz))
        Hmpo = H.to_mpo()
        result = dmrg(Hmpo, guess=self.random_uniform_mps(2, 3))
        self.assertAlmostEqual(result.energy, -2)
        self.assertAlmostEqual(Hmpo.expectation(result.state), -2)

    def test_dmrg_on_Heisenberg_five_sites(self):
        """Check we recover the exact ground state of a Heisenberg chain"""
        H = HeisenbergHamiltonian(size=5, field=[0.0, 0.0, 0.1])
        Hmpo = H.to_mpo()
        result = dmrg(Hmpo, guess=self.random_uniform_mps(2, 5))
        E, exact_v = scipy.sparse.linalg.eigsh(H.to_matrix(), k=1, which="SA")
        self.assertAlmostEqual(result.energy, E[0])
        v = result.state.to_vector()
        self.assertAlmostEqual(abs(np.vdot(v, exact_v)), np.float64(1.0))

    def test_dmrg_works_with_hamiltonian(self):
        """Check DMRG accepts an NNHamiltonian instead of an MPO"""
        H = ConstantTIHamiltonian(size=3, interaction=-np.kron(self.Sz, self.Sz))
        Hmpo = H.to_mpo()
        result = dmrg(H, guess=self.random_uniform_mps(2, 3))
        self.assertAlmostEqual(result.energy, -2)
        self.assertAlmostEqual(Hmpo.expectation(result.state), -2)

    def test_dmrg_uses_default_random_guess(self):
        """Check DMRG converges when no guess is provided (guess=None)."""
        H = ConstantTIHamiltonian(size=3, interaction=-np.kron(self.Sz, self.Sz))
        Hmpo = H.to_mpo()
        result = dmrg(Hmpo)
        self.assertAlmostEqual(result.energy, -2)

    # Starting canonical center / sweep direction
    def test_dmrg_uses_guess_with_canonical_centers_other_than_zero(self):
        """Check an interior canonical center is handled (re-centered to 0)."""
        H = ConstantTIHamiltonian(size=5, interaction=-np.kron(self.Sz, self.Sz))
        mps = CanonicalMPS(product_state([1.0, 1.0], 5), center=3, normalize=True)
        Hmpo = H.to_mpo()
        result = dmrg(Hmpo, guess=mps, maxiter=100)
        self.assertAlmostEqual(result.energy, -4)
        v = result.state.to_vector()
        self.assertAlmostEqual(v[0] ** 2 + v[-1] ** 2, 1.0)
        self.assertAlmostEqual(sum(v[1:-1] ** 2), 0.0)

    def test_dmrg_starts_from_right_edge_center(self):
        """A guess centered at the last site triggers a leftward first sweep."""
        size = 5
        H = ConstantTIHamiltonian(size=size, interaction=-np.kron(self.Sz, self.Sz))
        mps = CanonicalMPS(
            product_state([1.0, 1.0], size), center=size - 1, normalize=True
        )
        Hmpo = H.to_mpo()
        result = dmrg(Hmpo, guess=mps, maxiter=100)
        self.assertAlmostEqual(result.energy, -4)
        self.assertAlmostEqual(Hmpo.expectation(result.state).real, -4)

    # Variance and the returned state
    def test_dmrg_variance_vanishes_at_convergence(self):
        """A converged ground state must have ~zero energy variance."""
        H = HeisenbergHamiltonian(size=5, field=[0.0, 0.0, 0.1])
        Hmpo = H.to_mpo()
        result = dmrg(Hmpo, guess=self.random_uniform_mps(2, 5))
        self.assertLess(result.variances[-1], 1e-8)

    def test_dmrg_trajectory_and_variances_have_equal_length(self):
        H = HeisenbergHamiltonian(size=5, field=[0.0, 0.0, 0.1])
        Hmpo = H.to_mpo()
        result = dmrg(Hmpo, guess=self.random_uniform_mps(2, 5))
        self.assertEqual(len(result.trajectory), len(result.variances))

    def test_dmrg_returns_lowest_energy_visited(self):
        """`result.energy` is the minimum of the trajectory and matches the
        returned state's expectation value."""
        H = HeisenbergHamiltonian(size=6, field=[0.0, 0.0, 0.1])
        Hmpo = H.to_mpo()
        result = dmrg(Hmpo, guess=self.random_uniform_mps(2, 6))
        self.assertAlmostEqual(result.energy, min(result.trajectory))
        self.assertAlmostEqual(result.energy, Hmpo.expectation(result.state).real)

    def test_dmrg_energy_matches_returned_state_after_truncation(self):
        H = HeisenbergHamiltonian(size=6, field=[0.0, 0.0, 0.1])
        Hmpo = H.to_mpo()
        strategy = DEFAULT_STRATEGY.replace(max_bond_dimension=1)
        guess = product_state([1.0, 0.0], 6)

        result = dmrg(Hmpo, guess=guess, maxiter=5, strategy=strategy)

        self.assertAlmostEqual(result.energy, Hmpo.expectation(result.state).real)

    # Stopping conditions and argument validation
    def test_dmrg_honours_maxiter(self):
        """Exactly `maxiter` sweeps run, so the trajectory has maxiter+1 points."""
        H = HeisenbergHamiltonian(size=5, field=[0.0, 0.0, 0.1])
        Hmpo = H.to_mpo()
        result = dmrg(Hmpo, guess=self.random_uniform_mps(2, 5), maxiter=1)
        self.assertEqual(len(result.trajectory), 2)

    def test_dmrg_rejects_non_positive_maxiter(self):
        H = ConstantTIHamiltonian(size=3, interaction=-np.kron(self.Sz, self.Sz))
        with self.assertRaises(ValueError):
            dmrg(H.to_mpo(), maxiter=0)

    def test_dmrg_requires_at_least_two_sites(self):
        single_site = MPO([np.eye(2).reshape(1, 2, 2, 1)])
        with self.assertRaises(ValueError):
            dmrg(single_site)

    def test_dmrg_rejects_negative_tolerances(self):
        Hmpo = ConstantTIHamiltonian(
            size=3, interaction=-np.kron(self.Sz, self.Sz)
        ).to_mpo()
        with self.assertRaises(ValueError):
            dmrg(Hmpo, tol=-1.0)
        with self.assertRaises(ValueError):
            dmrg(Hmpo, tol_up=-1.0)


class TestDMRGConvergenceReason(SeeMPSTestCase):
    def test_large_upward_fluctuation_stops(self):
        reason = _convergence_reason(1.0, 1.0, tol=1e-10, tol_up=1e-10)
        self.assertIsNotNone(reason)

    def test_small_upward_fluctuation_continues(self):
        # Increase below tol_up: keep optimizing.
        reason = _convergence_reason(1e-4, 1.0, tol=1e-10, tol_up=1e-3)
        self.assertIsNone(reason)

    def test_slow_decrease_stops(self):
        reason = _convergence_reason(-1e-11, 1.0, tol=1e-10, tol_up=1e-10)
        self.assertIsNotNone(reason)

    def test_fast_decrease_continues(self):
        reason = _convergence_reason(-1.0, 1.0, tol=1e-10, tol_up=1e-10)
        self.assertIsNone(reason)

    def test_the_two_stopping_reasons_are_distinct(self):
        fluctuation = _convergence_reason(1.0, 1.0, tol=1e-10, tol_up=1e-10)
        slow = _convergence_reason(-1e-11, 1.0, tol=1e-10, tol_up=1e-10)
        self.assertNotEqual(fluctuation, slow)

    def test_energy_scale_is_applied(self):
        self.assertIsNotNone(
            _convergence_reason(-5.0, 100.0, tol=1e-1, tol_up=1e-1)
        )
        self.assertIsNone(_convergence_reason(-5.0, 1.0, tol=1e-1, tol_up=1e-1))
