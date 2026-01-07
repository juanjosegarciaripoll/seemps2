import numpy as np
from ..tools import TestCase
from seemps.state import MPS, CanonicalMPS, Strategy
from seemps.evolution.hdaf import split_step


class TestSplitStep(TestCase):
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
            itime=False,
        )
        assert isinstance(final_state, CanonicalMPS)
        self.assertIsInstance(final_state, CanonicalMPS)

        # Analytical solution
        A0 = 1.0 / (sigma0**2)
        ct = np.cos(final_time)
        st = np.sin(final_time)
        denom = ct + 1j * A0 * st
        At = (A0 * ct + 1j * st) / denom
        psi_exact = np.exp(-0.5 * At * x**2)
        psi_exact /= np.linalg.norm(psi_exact)

        overlap = abs(np.vdot(psi_exact, final_state.to_vector()))
        self.assertGreater(overlap, 0.999)

    def test_harmonic_oscillator_imaginary_time(self):
        num_qubits = 8
        dx = 0.1
        a = -dx * 2 ** (num_qubits - 1)
        periodic = True

        def potential_func(x):
            return 0.5 * x**2

        # Ground state to compare
        size = 2**num_qubits
        x = np.linspace(a, a + dx * size, size, endpoint=False)
        gs = np.exp(-0.5 * x**2)
        gs /= np.linalg.norm(gs)
        gs_mps = MPS.from_vector(gs, [2] * num_qubits)

        # Displaced initial state
        psi = np.exp(-0.5 * (x - 2.0) ** 2)
        psi /= np.linalg.norm(psi)
        state = MPS.from_vector(psi, [2] * num_qubits)

        strategy = Strategy(tolerance=1e-10)
        final_time = 5.0
        steps = 50

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
            itime=True,
        )
        assert isinstance(final_state, CanonicalMPS)
        self.assertIsInstance(final_state, CanonicalMPS)

        final_state = final_state.normalize_inplace()
        overlap = abs(final_state.to_vector() @ gs_mps.to_vector())
        self.assertGreater(overlap, 0.999)
