import numpy as np
from ..tools import TestCase, random_uniform_mps
from seemps.analysis.operators import (
    id_mpo,
    x_mpo,
    p_mpo,
    exponential_mpo,
    cos_mpo,
    sin_mpo,
)


class Test_analysis_operators(TestCase):
    n_qubits = 6
    N = 2**n_qubits
    L = 10
    a = -L / 2
    dx = L / N
    x = a + dx * np.arange(N)
    k = 2 * np.pi * np.arange(N) / L
    p = k - (np.arange(N) >= (N / 2)) * 2 * np.pi / dx
    f = random_uniform_mps(2, n_qubits)

    def test_id_mpo(self):
        self.assertSimilar(self.f, id_mpo(self.n_qubits) @ self.f)

    def test_x_mpo(self):
        self.assertSimilar(
            self.x * self.f.to_vector(), x_mpo(self.n_qubits, self.a, self.dx) @ self.f
        )

    def test_x_n_mpo(self):
        n = 3
        self.assertSimilar(
            self.x**n * self.f.to_vector(),
            (x_mpo(self.n_qubits, self.a, self.dx) ** n) @ self.f,
        )

    def test_p_mpo(self):
        self.assertSimilar(
            self.p * self.f.to_vector(), p_mpo(self.n_qubits, self.dx) @ self.f
        )

    def test_p_n_mpo(self):
        n = 3
        self.assertSimilar(
            self.p**n * self.f.to_vector(),
            (p_mpo(self.n_qubits, self.dx) ** n) @ self.f,
        )

    def test_exp_mpo(self):
        c = -2j
        self.assertSimilar(
            np.exp(c * self.x) * self.f.to_vector(),
            exponential_mpo(self.n_qubits, self.a, self.dx, c) @ self.f,
        )

    def test_cos_mpo(self):
        self.assertSimilar(
            np.cos(self.x) * self.f.to_vector(),
            cos_mpo(self.n_qubits, self.a, self.dx) @ self.f,
        )

    def test_sin_mpo(self):
        self.assertSimilar(
            np.sin(self.x) * self.f.to_vector(),
            sin_mpo(self.n_qubits, self.a, self.dx) @ self.f,
        )
