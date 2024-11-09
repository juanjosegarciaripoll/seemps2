from ..tools import TestCase

import numpy as np

from seemps.state import MPS
from seemps.analysis.hdaf import hdaf_mpo


# Test params
qubit_range = range(6, 10 + 1)
space_domain = (-8, 8)
propagator_time = 0.1


def gaussian(x):
    state = np.exp(- 0.5 * x ** 2)
    return state / np.linalg.norm(state)


def gaussian_deriv(x):
    return - x * gaussian(x)


def gaussian_deriv2(x):
    return (x ** 2 - 1) * gaussian(x)


class TestHDAF(TestCase):
    def test_identity(self):
        a, b = space_domain
        for nq in qubit_range:
            x, dx = np.linspace(a, b, num=2**nq, retstep=True, endpoint=False)
            state = MPS.from_vector(gaussian(x), [2]*nq)
            hdaf = hdaf_mpo(num_qubits=nq, dx=dx, M=20)

            self.assertSimilarStates(state, hdaf @ state)

    def test_derivative(self):
        a, b = space_domain
        for nq in qubit_range:
            x, dx = np.linspace(a, b, num=2**nq, retstep=True, endpoint=False)
            state = MPS.from_vector(gaussian(x), [2]*nq)
            deriv = MPS.from_vector(gaussian_deriv(x), [2]*nq)
            hdaf = hdaf_mpo(num_qubits=nq, dx=dx, M=20, derivative=1)

            self.assertSimilarStates(deriv, hdaf @ state)

    def test_derivative_2(self):
        a, b = space_domain
        for nq in qubit_range:
            x, dx = np.linspace(a, b, num=2**nq, retstep=True, endpoint=False)
            state = MPS.from_vector(gaussian(x), [2]*nq)
            deriv = MPS.from_vector(gaussian_deriv2(x), [2]*nq)
            hdaf = hdaf_mpo(num_qubits=nq, dx=dx, M=20, derivative=2)

            self.assertSimilarStates(deriv, hdaf @ state)

    def test_free_propagator(self, time=propagator_time):
        a, b = space_domain
        st = np.sqrt(1 + 1j * time)
        for nq in qubit_range:
            x, dx = np.linspace(a, b, num=2**nq, retstep=True, endpoint=False)
            state = MPS.from_vector(gaussian(x), [2]*nq)
            evol = MPS.from_vector(gaussian(x / st) / st, [2]*nq)
            hdaf = hdaf_mpo(num_qubits=nq, dx=dx, M=20, time=time)

            self.assertSimilarStates(evol, hdaf @ state)
