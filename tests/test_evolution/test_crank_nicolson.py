import numpy as np
from seemps.state import CanonicalMPS, DEFAULT_STRATEGY, product_state
from seemps.operators import MPO
from seemps.evolution.crank_nicolson import crank_nicolson
from seemps.hamiltonians import HeisenbergHamiltonian
from .problem import EvolutionTestCase


class TestCrankNicolson(EvolutionTestCase):
    def test_crank_nicolson_time_steps_and_callback(self):
        """Check the integration times used by the algorithm"""
        nqubits = 4
        mps = product_state([np.ones(2) / np.sqrt(2)] * nqubits)
        H = HeisenbergHamiltonian(nqubits).to_mpo()

        final = crank_nicolson(H, 1.0, mps, steps=10, callback=lambda t, state: t)
        self.assertSimilar(final, np.linspace(0, 1.0, 11))

        final = crank_nicolson(H, (1.0, 2.0), mps, steps=10, callback=lambda t, state: t)
        self.assertSimilar(final, np.linspace(1.0, 2.0, 11))

        t_span = np.linspace(1.0, 3.0, 13)
        final = crank_nicolson(H, t_span, mps, steps=10, callback=lambda t, state: t)
        self.assertSimilar(final, t_span)

    def test_crank_nicolson_accumulated_phase(self):
        """Evolve with a state that is invariant under the Hamiltonian
        and check the accumulated phase."""
        T = 0.01
        steps = 1
        dt = T / steps
        nqubits = 4
        mps = product_state([np.ones(2) / np.sqrt(2)] * nqubits)

        H = HeisenbergHamiltonian(nqubits).to_mpo()
        final = crank_nicolson(H, T, mps, steps=steps)
        self.assertSimilarStates(final, mps)

        E = H.expectation(mps)
        phase = np.exp(-1j * dt * E * steps)
        self.assertSimilar(final, phase * mps)
