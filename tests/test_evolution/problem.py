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
from ..tools import SeeMPSTestCase


class EvolutionTestCase(SeeMPSTestCase):
    Heisenberg2 = 0.25 * (np.kron(σx, σx) + np.kron(σy, σy) + np.kron(σz, σz))

    def random_initial_state(self, size: int) -> MPS:
        states = self.rng.normal(size=(size, 2)) + 1j * self.rng.normal(size=(size, 2))
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
    def solve_ode(
        self,
        L: MPO,
        time: TimeSpan,
        state: MPS,
        steps: int = 1000,
        strategy: Strategy = DEFAULT_STRATEGY,
        callback: ODECallback | None = None,
    ) -> MPS | list[Any]:
        pass

    def test_time_steps_and_callback(self):
        """Check the integration times used by the algorithm"""
        nqubits = 4
        mps = product_state([np.ones(2) / sqrt(2)] * nqubits)
        L = HeisenbergHamiltonian(nqubits).to_mpo()

        final = self.solve_ode(L, 1.0, mps, steps=10, callback=lambda t, state: t)
        self.assertSimilar(final, np.linspace(0, 1.0, 11))

        final = self.solve_ode(
            L, (1.0, 2.0), mps, steps=10, callback=lambda t, state: t
        )
        self.assertSimilar(final, np.linspace(1.0, 2.0, 11))

        t_span = np.linspace(1.0, 3.0, 13)
        final = self.solve_ode(L, t_span, mps, steps=10, callback=lambda t, state: t)
        self.assertSimilar(final, t_span)

    def test_accumulated_amplification(self):
        """Evolve an eigenstate and check the accumulated amplification."""
        T = 0.01
        steps = 1
        dt = T / steps
        nqubits = 4
        mps = product_state([np.ones(2) / sqrt(2)] * nqubits)

        L = HeisenbergHamiltonian(nqubits).to_mpo()
        final = self.solve_ode(L, T, mps, steps=steps)
        self.assertSimilarStates(final, mps)

        E = L.expectation(mps)
        amplification = self.accumulated_amplification(E, dt, steps)
        self.assertIsInstance(final, MPS)
        self.assertSimilar(final, amplification * mps)

    def accumulated_amplification(self, E, dt, steps):
        return np.exp(dt * steps * E)
