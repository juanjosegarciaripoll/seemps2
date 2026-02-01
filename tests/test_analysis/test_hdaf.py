from ..tools import SeeMPSTestCase
import numpy as np
from seemps.state import MPS
from seemps.analysis.hdaf import hdaf_mpo
from seemps.analysis.derivatives import hdaf_derivative_mpo
from seemps.analysis.mesh import QuantizedInterval


def gaussian(x):
    state = np.exp(-0.5 * x**2)
    return state / np.linalg.norm(state)


def gaussian_deriv(x):
    return -x * gaussian(x)


def gaussian_deriv2(x):
    return (x**2 - 1) * gaussian(x)


class TestHDAF(SeeMPSTestCase):
    def setUp(self) -> None:
        self.qubit_range = range(6, 10 + 1)
        self.space_domain = (-8, 8)
        self.propagator_time = 0.1

    def test_identity(self):
        a, b = self.space_domain
        for nq in self.qubit_range:
            with self.subTest(n_qubits=nq):
                interval = QuantizedInterval(a, b, nq, endpoint_right=False)
                x = interval.to_vector()
                state = MPS.from_vector(gaussian(x), [2] * nq)
                hdaf = hdaf_derivative_mpo(order=0, interval=interval, M=20)
                self.assertSimilarStates(state, hdaf @ state)

    def test_derivative(self):
        a, b = self.space_domain
        for nq in self.qubit_range:
            with self.subTest(n_qubits=nq):
                interval = QuantizedInterval(a, b, nq, endpoint_right=False)
                x = interval.to_vector()
                state = MPS.from_vector(gaussian(x), [2] * nq)
                deriv = MPS.from_vector(gaussian_deriv(x), [2] * nq)
                hdaf = hdaf_derivative_mpo(order=1, interval=interval, M=20)

                self.assertSimilarStates(deriv, hdaf @ state)

    def test_derivative_2(self):
        a, b = self.space_domain
        for nq in self.qubit_range:
            with self.subTest(n_qubits=nq):
                interval = QuantizedInterval(a, b, nq, endpoint_right=False)
                x = interval.to_vector()
                state = MPS.from_vector(gaussian(x), [2] * nq)
                deriv = MPS.from_vector(gaussian_deriv2(x), [2] * nq)
                hdaf = hdaf_derivative_mpo(order=2, interval=interval, M=20)

                self.assertSimilarStates(deriv, hdaf @ state)

    def test_free_propagator(self):
        time = self.propagator_time
        a, b = self.space_domain
        st = np.sqrt(1 + 1j * time)
        for nq in self.qubit_range:
            x, dx = np.linspace(a, b, num=2**nq, retstep=True, endpoint=False)
            state = MPS.from_vector(gaussian(x), [2] * nq)
            evol = MPS.from_vector(gaussian(x / st) / st, [2] * nq)
            hdaf = hdaf_mpo(num_qubits=nq, dx=dx, M=20, time=time)

            self.assertSimilarStates(evol, hdaf @ state)
