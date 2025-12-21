from abc import abstractmethod
from typing import Any
import numpy as np
import unittest
from math import sqrt
from seemps.tools import σx, σy, σz
from seemps.state import product_state, MPS, DEFAULT_STRATEGY, Strategy
from seemps.operators import MPO
from seemps.hamiltonians import HeisenbergHamiltonian
from seemps.evolution import ODECallback, TimeSpan
from ..tools import TestCase


class EvolutionTestCase(TestCase):
    Heisenberg2 = 0.25 * (np.kron(σx, σx) + np.kron(σy, σy) + np.kron(σz, σz))

    def random_initial_state(self, size: int) -> MPS:
        states = np.random.randn(size, 2) + 1j * np.random.randn(size, 2)
        for n in range(size):
            states[n, :] /= np.linalg.norm(states[n, :])
        return product_state(states)


class RKTypeEvolutionTestcase(EvolutionTestCase):
    tolerance: float = 1e-10
    name: str = "Unknown"

    @classmethod
    def setUpClass(cls):
        if cls is RKTypeEvolutionTestcase:
            raise unittest.SkipTest(f"Skip {cls} tests, it's a base class")
        super().setUpClass()

    @abstractmethod
    def solve_Schroedinger(
        self,
        H: MPO,
        time: TimeSpan,
        state: MPS,
        steps: int = 1000,
        strategy: Strategy = DEFAULT_STRATEGY,
        callback: ODECallback | None = None,
        itime: bool = False,
    ) -> MPS | list[Any]:
        pass

    def test_time_steps_and_callback(self):
        """Check the integration times used by the algorithm"""
        nqubits = 4
        mps = product_state([np.ones(2) / sqrt(2)] * nqubits)
        H = HeisenbergHamiltonian(nqubits).to_mpo()

        final = self.solve_Schroedinger(
            H, 1.0, mps, steps=10, callback=lambda t, state: t
        )
        self.assertSimilar(final, np.linspace(0, 1.0, 11))

        final = self.solve_Schroedinger(
            H, (1.0, 2.0), mps, steps=10, callback=lambda t, state: t
        )
        self.assertSimilar(final, np.linspace(1.0, 2.0, 11))

        t_span = np.linspace(1.0, 3.0, 13)
        final = self.solve_Schroedinger(
            H, t_span, mps, steps=10, callback=lambda t, state: t
        )
        self.assertSimilar(final, t_span)

    def test_accumulated_phase(self):
        """Evolve with a state that is invariant under the Hamiltonian
        and check the accumulated phase."""
        T = 0.01
        steps = 1
        dt = T / steps
        nqubits = 4
        mps = product_state([np.ones(2) / sqrt(2)] * nqubits)

        H = HeisenbergHamiltonian(nqubits).to_mpo()
        final = self.solve_Schroedinger(H, T, mps, steps=steps)
        self.assertSimilarStates(final, mps)

        E = H.expectation(mps)
        phase = self.accummulated_phase(E, dt, steps)
        self.assertIsInstance(final, MPS)
        self.assertSimilar(final, phase * mps)

    def accummulated_phase(self, E, dt, steps):
        return np.exp(-1j * dt * steps * E)
