import numpy as np
from ..tools import SeeMPSTestCase
from seemps.state import MPS, CanonicalMPS, Strategy
from seemps.evolution.hdaf import split_step


class TestSplitStep(SeeMPSTestCase):
    def test_analytical_evolution(self):
        num_qubits = 8
        dx = 0.1
        a = -dx * 2 ** (num_qubits - 1)
        periodic = True

        def potential_func(x):
            return 0.5 * x**2

        sigma0 = 0.5
        size = 2**num_qubits
        x = np.linspace(a, a + dx * size, size, endpoint=False)
        psi = np.exp(-0.5 * (x / sigma0) ** 2)
        psi /= np.linalg.norm(psi)
        state = MPS.from_vector(psi, [2] * num_qubits)

        strategy = Strategy(tolerance=1e-10)
        final_time = 2.0
        steps = 20

        final_state = split_step(
            potential_func=potential_func,
            time=final_time,
            state=state,
            a=a,
            num_qubits=num_qubits,
            dx=dx,
            periodic=periodic,
            steps=steps,
            strategy=strategy,
        )
        self.assertIsInstance(final_state, CanonicalMPS)

        A0 = 1.0 / sigma0**2
        ct, st = np.cos(final_time), np.sin(final_time)
        At = (A0 * ct + 1j * st) / (ct + 1j * A0 * st)
        psi_exact = np.exp(-0.5 * At * x**2)
        psi_exact /= np.linalg.norm(psi_exact)

        final_vec = final_state.to_vector()
        final_vec /= np.linalg.norm(final_vec)
        self.assertGreater(abs(np.vdot(psi_exact, final_vec)), 0.999)

    def test_coherent_state_evolution(self):
        # A displaced HO ground state (coherent state) has an exact solution:
        # psi(x,t) ~ exp(i*x0*sin(t)*x) * exp(-0.5*(x - x0*cos(t))^2)
        # The center oscillates at x0*cos(t) and a momentum kick builds up.
        num_qubits = 8
        dx = 0.1
        a = -dx * 2 ** (num_qubits - 1)
        periodic = True

        def potential_func(x):
            return 0.5 * x**2

        x0 = 2.0
        size = 2**num_qubits
        x = np.linspace(a, a + dx * size, size, endpoint=False)
        psi = np.exp(-0.5 * (x - x0) ** 2)
        psi /= np.linalg.norm(psi)
        state = MPS.from_vector(psi, [2] * num_qubits)

        strategy = Strategy(tolerance=1e-10)
        final_time = 2.0
        steps = 20

        final_state = split_step(
            potential_func=potential_func,
            time=final_time,
            state=state,
            a=a,
            num_qubits=num_qubits,
            dx=dx,
            periodic=periodic,
            steps=steps,
            strategy=strategy,
        )
        self.assertIsInstance(final_state, CanonicalMPS)

        ct, st = np.cos(final_time), np.sin(final_time)
        psi_exact = np.exp(-1j * x0 * st * x) * np.exp(-0.5 * (x - x0 * ct) ** 2)
        psi_exact /= np.linalg.norm(psi_exact)

        final_vec = final_state.to_vector()
        final_vec /= np.linalg.norm(final_vec)
        self.assertGreater(abs(np.vdot(psi_exact, final_vec)), 0.999)
