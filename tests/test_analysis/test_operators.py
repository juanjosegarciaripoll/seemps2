import numpy as np
from ..tools import SeeMPSTestCase
from seemps.analysis.operators import (
    id_mpo,
    x_mpo,
    x_to_n_mpo,
    p_mpo,
    p_to_n_mpo,
    exponential_mpo,
    cos_mpo,
    sin_mpo,
    mpo_affine,
    mpo_cumsum,
)
from seemps.state import MPS


class Test_analysis_operators(SeeMPSTestCase):
    n_qubits = 6
    N = 2**n_qubits
    L = 10
    a = -L / 2
    dx = L / N
    x = a + dx * np.arange(N)
    k = 2 * np.pi * np.arange(N) / L
    p = k - (np.arange(N) >= (N / 2)) * 2 * np.pi / dx

    def setUp(self):
        super().setUp()
        self.f = self.random_uniform_mps(2, self.n_qubits)

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

    def test_small_dense_reference_operators_for_one_qubit(self):
        a = -0.5
        dx = 0.25
        self.assertSimilar(x_mpo(1, a, dx).to_matrix(), np.diag([a, a + dx]))

        dk = 2 * np.pi / (dx * 2)
        self.assertSimilar(p_mpo(1, dx).to_matrix(), np.diag([0.0, -dk]))

        c = 1.0 + 0.5j
        self.assertSimilar(
            exponential_mpo(1, a, dx, c).to_matrix(),
            np.diag([np.exp(c * a), np.exp(c * (a + dx))]),
        )

    def test_x_to_n_and_p_to_n_dense_reference_on_small_system(self):
        n_qubits = 2
        a = -1.0
        dx = 0.5
        x = a + dx * np.arange(2**n_qubits)
        k = 2 * np.pi * np.arange(2**n_qubits) / (dx * 2**n_qubits)
        p = k - (np.arange(2**n_qubits) >= (2 ** (n_qubits - 1))) * 2 * np.pi / dx

        self.assertSimilar(x_to_n_mpo(n_qubits, a, dx, 3).to_matrix(), np.diag(x**3))
        self.assertSimilar(p_to_n_mpo(n_qubits, dx, 2).to_matrix(), np.diag(p**2))

    def test_trigonometric_mpos_match_dense_reference_on_small_system(self):
        n_qubits = 2
        a = -1.0
        dx = 0.5
        x = a + dx * np.arange(2**n_qubits)
        self.assertSimilar(cos_mpo(n_qubits, a, dx).to_matrix(), np.diag(np.cos(x)))
        self.assertSimilar(sin_mpo(n_qubits, a, dx).to_matrix(), np.diag(np.sin(x)))

    def test_mpo_affine_handles_zero_and_nonzero_offset(self):
        base = x_mpo(2, -1.0, 0.5)
        self.assertSimilar(
            mpo_affine(base, (-1.0, 0.5), (2.0, -1.0)).to_matrix(),
            -2.0 * base.to_matrix(),
        )
        self.assertSimilar(
            mpo_affine(base, (-1.0, 0.5), (0.0, 1.5)).to_matrix(),
            base.to_matrix() + np.eye(base.to_matrix().shape[0]),
        )

    def test_mpo_cumsum_dense_reference(self):
        state = MPS.from_vector(np.array([1.0, 2.0, 3.0, 4.0]), [2, 2], normalize=False)
        expected = np.array([1.0, 3.0, 6.0, 10.0])
        self.assertSimilar(mpo_cumsum(2) @ state, expected)

        state3 = MPS.from_vector(np.arange(1.0, 9.0), [2, 2, 2], normalize=False)
        expected3 = np.cumsum(np.arange(1.0, 9.0))
        self.assertSimilar(mpo_cumsum(3) @ state3, expected3)
