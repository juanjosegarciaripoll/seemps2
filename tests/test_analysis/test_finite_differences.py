from typing import cast
import numpy as np
from scipy.sparse import spdiags, csr_matrix, eye
from seemps.analysis.finite_differences import smooth_finite_differences_mpo
from .. import tools


class TestFiniteDifferences(tools.TestCase):
    def Down(self, nqubits: int, periodic: bool = False) -> csr_matrix:
        """Moves f[i] to f[i-1]"""
        L = 2**nqubits
        if periodic:
            M = spdiags([np.ones(L), np.ones(L)], [1, -L + 1], format="csr")  # type: ignore # scipy-stubs broken
        else:
            M = spdiags([np.ones(L)], [1], format="csr")  # type: ignore # scipy-stubs broken
        return cast(csr_matrix, M)

    def test_first_derivative_two_qubits_perodic(self):
        dx = 0.1
        D2 = smooth_finite_differences_mpo(
            2, order=1, filter=3, periodic=True, dx=dx
        ).to_matrix()
        self.assertSimilar(
            D2,
            [
                [0, 0.5 / dx, 0, -0.5 / dx],
                [-0.5 / dx, 0, 0.5 / dx, 0],
                [0, -0.5 / dx, 0, 0.5 / dx],
                [0.5 / dx, 0, -0.5 / dx, 0],
            ],
        )
        D = self.Down(2, periodic=True)
        self.assertSimilar(D2, (D - D.T) / (2.0 * dx))

    def test_second_derivative_two_qubits_perodic(self):
        dx = 0.1
        D2 = smooth_finite_differences_mpo(
            2, order=2, filter=3, periodic=True, dx=dx
        ).to_matrix()
        dx2 = dx * dx
        self.assertSimilar(
            D2,
            [
                [-2 / dx2, 1 / dx2, 0, 1 / dx2],
                [1 / dx2, -2 / dx2, 1 / dx2, 0],
                [0, 1 / dx2, -2 / dx2, 1 / dx2],
                [1 / dx2, 0, 1 / dx2, -2 / dx2],
            ],
        )
        D = self.Down(2, periodic=True)
        I = eye(4)
        self.assertSimilar(D2, (D - 2 * I + D.T) / dx2)

    def test_first_derivative_two_qubits_non_perodic(self):
        dx = 0.1
        D2 = smooth_finite_differences_mpo(
            2, order=1, filter=3, periodic=False, dx=dx
        ).to_matrix()
        self.assertSimilar(
            D2,
            [
                [0, 0.5 / dx, 0, 0],
                [-0.5 / dx, 0, 0.5 / dx, 0],
                [0, -0.5 / dx, 0, 0.5 / dx],
                [0, 0, -0.5 / dx, 0],
            ],
        )
        D = self.Down(2, periodic=False)
        self.assertSimilar(D2, (D - D.T) / (2.0 * dx))

    def test_second_derivative_two_qubits_non_perodic(self):
        dx = 0.1
        D2 = smooth_finite_differences_mpo(
            2, order=2, filter=3, periodic=False, dx=dx
        ).to_matrix()
        dx2 = dx * dx
        self.assertSimilar(
            D2,
            [
                [-2 / dx2, 1 / dx2, 0, 0],
                [1 / dx2, -2 / dx2, 1 / dx2, 0],
                [0, 1 / dx2, -2 / dx2, 1 / dx2],
                [0, 0, 1 / dx2, -2 / dx2],
            ],
        )
        D = self.Down(2, periodic=False)
        I = eye(4)
        self.assertSimilar(D2, (D - 2 * I + D.T) / dx2)
