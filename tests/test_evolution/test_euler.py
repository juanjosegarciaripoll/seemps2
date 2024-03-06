import numpy as np

from seemps.evolution.euler import euler, euler2, implicit_euler
from seemps.hamiltonians import HeisenbergHamiltonian
from seemps.operators import MPO
from seemps.state import DEFAULT_STRATEGY, CanonicalMPS, product_state

from .problem import EvolutionTestCase


class TestEuler(EvolutionTestCase):
    def test_euler_time_steps_and_callback(self):
        """Check the integration times used by the algorithm"""
        nqubits = 4
        mps = product_state([np.ones(2) / np.sqrt(2)] * nqubits)
        H = HeisenbergHamiltonian(nqubits).to_mpo()

        final = euler(H, 1.0, mps, steps=10, callback=lambda t, state: t)
        self.assertSimilar(final, np.linspace(0, 1.0, 11))

        final = euler(H, (1.0, 2.0), mps, steps=10, callback=lambda t, state: t)
        self.assertSimilar(final, np.linspace(1.0, 2.0, 11))

        t_span = np.linspace(1.0, 3.0, 13)
        final = euler(H, t_span, mps, steps=10, callback=lambda t, state: t)
        self.assertSimilar(final, t_span)

    def test_euler_accumulated_phase(self):
        """Evolve with a state that is invariant under the Hamiltonian
        and check the accumulated phase."""
        T = 0.01
        steps = 1
        dt = T / steps
        nqubits = 4
        mps = product_state([np.ones(2) / np.sqrt(2)] * nqubits)

        H = HeisenbergHamiltonian(nqubits).to_mpo()
        final = euler(H, T, mps, steps=steps)
        self.assertSimilarStates(final, mps)

        E = H.expectation(mps)
        phase = (1 - 1j * E * dt) ** steps
        self.assertSimilar(final, phase * mps)


class TestEuler2(EvolutionTestCase):
    def test_euler2_time_steps_and_callback(self):
        """Check the integration times used by the algorithm"""
        nqubits = 4
        mps = product_state([np.ones(2) / np.sqrt(2)] * nqubits)
        H = HeisenbergHamiltonian(nqubits).to_mpo()

        final = euler2(H, 1.0, mps, steps=10, callback=lambda t, state: t)
        self.assertSimilar(final, np.linspace(0, 1.0, 11))

        final = euler2(H, (1.0, 2.0), mps, steps=10, callback=lambda t, state: t)
        self.assertSimilar(final, np.linspace(1.0, 2.0, 11))

        t_span = np.linspace(1.0, 3.0, 13)
        final = euler2(H, t_span, mps, steps=10, callback=lambda t, state: t)
        self.assertSimilar(final, t_span)

    def test_euler2_accumulated_phase(self):
        """Evolve with a state that is invariant under the Hamiltonian
        and check the accumulated phase."""
        T = 0.01
        steps = 1
        dt = T / steps
        nqubits = 4
        mps = product_state([np.ones(2) / np.sqrt(2)] * nqubits)

        H = HeisenbergHamiltonian(nqubits).to_mpo()
        final = euler2(H, T, mps, steps=steps)
        self.assertSimilarStates(final, mps)

        E = H.expectation(mps)
        phase = 1 - 0.5j * dt * E * (2.0 - 1j * E * dt)
        self.assertSimilar(final, phase * mps)


class TestImplicitEuler(EvolutionTestCase):
    def test_implicit_euler_time_steps_and_callback(self):
        """Check the integration times used by the algorithm"""
        nqubits = 4
        mps = product_state([np.ones(2) / np.sqrt(2)] * nqubits)
        H = HeisenbergHamiltonian(nqubits).to_mpo()

        final = implicit_euler(H, 1.0, mps, steps=10, callback=lambda t, state: t)
        self.assertSimilar(final, np.linspace(0, 1.0, 11))

        final = implicit_euler(
            H, (1.0, 2.0), mps, steps=10, callback=lambda t, state: t
        )
        self.assertSimilar(final, np.linspace(1.0, 2.0, 11))

        t_span = np.linspace(1.0, 3.0, 13)
        final = implicit_euler(H, t_span, mps, steps=10, callback=lambda t, state: t)
        self.assertSimilar(final, t_span)

    def test_implicit_euler_accumulated_phase(self):
        """Evolve with a state that is invariant under the Hamiltonian
        and check the accumulated phase."""
        T = 0.01
        steps = 1
        dt = T / steps
        nqubits = 4
        mps = product_state([np.ones(2) / np.sqrt(2)] * nqubits)

        H = HeisenbergHamiltonian(nqubits).to_mpo()
        final = implicit_euler(H, T, mps, steps=steps)
        self.assertSimilarStates(final, mps)

        E = H.expectation(mps)
        phase = 1 - 0.5j * dt * E * (2.0 - 1j * E * dt)
        self.assertSimilar(final, phase * mps)
