from typing import Any

import numpy as np
import scipy.linalg

from seemps.evolution import ODECallback, TimeSpan, gausslegendre
from seemps.evolution.gausslegendre import _gl_tableau, gausslegendre_step
from seemps.hamiltonians import HeisenbergHamiltonian
from seemps.operators import MPO
from seemps.operators.projectors import identity_mpo
from seemps.state import DEFAULT_STRATEGY, MPS, Strategy

from .problem import EvolutionTestCase, RKTypeEvolutionTestcase


def stability_function(z: complex, stages: int) -> complex:
    """R(z) = 1 + z b^T (I - z A)^{-1} 1, the Pade approximant of exp(z)."""
    _, A, b, _ = _gl_tableau(stages)
    ones = np.ones(stages)
    return 1.0 + z * (b @ np.linalg.solve(np.eye(stages) - z * A, ones))


class TestGaussLegendre(RKTypeEvolutionTestcase):
    stages: int = 2

    def solve_ode(
        self,
        L: MPO,
        time: TimeSpan,
        state: MPS,
        steps: int = 1000,
        strategy: Strategy = DEFAULT_STRATEGY,
        callback: ODECallback | None = None,
    ) -> MPS | list[Any]:
        return gausslegendre(
            L,
            time,
            state,
            steps=steps,
            stages=self.stages,
            strategy=strategy,
            callback=callback,
        )

    def accumulated_amplification(self, E, dt, steps):
        # A Gauss-Legendre step amplifies an eigenstate by the (s, s) Pade
        # approximant of the exponential, not by the exponential itself.
        return stability_function(dt * E, self.stages) ** steps


class TestGaussLegendre3Stages(TestGaussLegendre):
    stages = 3


class TestGaussLegendreLinear(EvolutionTestCase):
    """The linear branch, checked against dense algebra."""

    # The stage solves must not be what limits the accuracy under test, so
    # these cases pin `rtol` well below the discretization errors they measure,
    # instead of relying on the default meant for production runs.
    RTOL = 1e-12

    def tight_strategy(self) -> Strategy:
        return DEFAULT_STRATEGY.replace(
            normalize=False, tolerance=1e-18, simplification_tolerance=1e-18
        )

    def test_matches_matrix_exponential(self):
        """A full evolution reproduces expm(T*L) applied to the initial state."""
        nqubits = 4
        T = 0.2
        strategy = self.tight_strategy()
        state = self.random_initial_state(nqubits)
        L = (-1j) * HeisenbergHamiltonian(nqubits).to_mpo()
        exact = scipy.linalg.expm(T * L.to_matrix()) @ state.to_vector()

        for stages in (1, 2, 3):
            with self.subTest(stages=stages):
                final = gausslegendre(
                    L,
                    T,
                    state,
                    steps=20,
                    stages=stages,
                    strategy=strategy,
                    rtol=self.RTOL,
                )
                self.assertIsInstance(final, MPS)
                self.assertSimilar(final, exact)

    def test_order_of_convergence(self):
        """The local error of the s-stage method scales as dt^(2s)."""
        nqubits = 3
        T = 0.5
        strategy = self.tight_strategy()
        state = self.random_initial_state(nqubits)
        L = (-1j) * HeisenbergHamiltonian(nqubits).to_mpo()
        exact = scipy.linalg.expm(T * L.to_matrix()) @ state.to_vector()

        # Only the low orders are checked: with `stages = 3` the error is
        # already at the 1e-10 truncation floor, where the rate is meaningless.
        for stages in (1, 2):
            with self.subTest(stages=stages):
                errors = []
                for steps in (2, 4):
                    final = gausslegendre(
                        L,
                        T,
                        state,
                        steps=steps,
                        stages=stages,
                        strategy=strategy,
                        rtol=self.RTOL,
                    )
                    assert isinstance(final, MPS)
                    errors.append(np.linalg.norm(final.to_vector() - exact))
                # Halving dt must reduce the error by ~2^(2*stages).
                self.assertGreater(errors[0] / errors[1], 0.8 * 2 ** (2 * stages))

    def test_rtol_controls_the_stage_solves(self):
        """An explicit `rtol` reaches the linear solver."""
        nqubits = 4
        T = 0.2
        strategy = self.tight_strategy()
        state = self.random_initial_state(nqubits)
        L = (-1j) * HeisenbergHamiltonian(nqubits).to_mpo()
        exact = scipy.linalg.expm(T * L.to_matrix()) @ state.to_vector()

        def error(rtol):
            final = gausslegendre(L, T, state, steps=4, strategy=strategy, rtol=rtol)
            assert isinstance(final, MPS)
            return np.linalg.norm(final.to_vector() - exact)

        # Under-solving the stages must show up as a worse answer.
        self.assertGreater(error(1e-2), 100 * error(self.RTOL))

    def test_rejects_zero_stages(self):
        state = self.random_initial_state(2)
        L = (-1j) * HeisenbergHamiltonian(2).to_mpo()
        with self.assertRaises(ValueError):
            gausslegendre_step(L, 0.0, state, 0.1, stages=0)


class TestGaussLegendreNonlinear(EvolutionTestCase):
    """The Gauss-Seidel branch driven by a state-dependent term N."""

    # The stage solves must not be what limits the accuracy under test, so
    # these cases pin `rtol` well below the discretization errors they measure,
    # instead of relying on the default meant for production runs.
    RTOL = 1e-12

    def tight_strategy(self) -> Strategy:
        return DEFAULT_STRATEGY.replace(
            normalize=False, tolerance=1e-18, simplification_tolerance=1e-18
        )

    def test_constant_N_matches_linear_branch(self):
        """A state-independent N must give the same answer as folding it into L."""
        nqubits = 3
        T = 0.2
        strategy = self.tight_strategy()
        state = self.random_initial_state(nqubits)
        H = HeisenbergHamiltonian(nqubits).to_mpo()
        L = (-1j) * H
        M = (-0.5j) * H

        linear = gausslegendre(
            (-1.5j) * H, T, state, steps=4, strategy=strategy, rtol=self.RTOL
        )
        nonlinear = gausslegendre(
            L,
            T,
            state,
            steps=4,
            strategy=strategy,
            rtol=self.RTOL,
            N=lambda t, psi: M,
        )
        self.assertIsInstance(nonlinear, MPS)
        self.assertSimilar(nonlinear, linear)

    def test_norm_follows_the_exact_solution(self):
        r"""Integrate d|psi>/dt = g ||psi||^2 |psi>, whose norm is known exactly.

        With `L = 0` and `N = g ||psi||^2 I`, the norm obeys `dn/dt = g n^3`,
        so `n(t) = n0 / sqrt(1 - 2 g n0^2 t)`.
        """
        nqubits = 3
        g = 0.3
        T = 0.4
        strategy = self.tight_strategy()
        state = self.random_initial_state(nqubits)
        identity = identity_mpo([2] * nqubits)
        zero = 0.0 * identity

        n0 = state.norm()
        final = gausslegendre(
            zero,
            T,
            state,
            steps=8,
            strategy=strategy,
            rtol=self.RTOL,
            N=lambda t, psi: (g * psi.norm_squared()) * identity,
        )
        assert isinstance(final, MPS)
        expected = n0 / np.sqrt(1.0 - 2.0 * g * n0**2 * T)
        self.assertAlmostEqual(final.norm(), expected, places=6)
