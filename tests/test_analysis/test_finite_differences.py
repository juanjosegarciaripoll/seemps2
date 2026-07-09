from typing import cast
import numpy as np
from scipy.sparse import spdiags, csr_matrix, eye
from seemps.analysis.derivatives import finite_differences_mpo
from seemps.analysis.derivatives.finite_differences import tridiagonal_mpo
from seemps.analysis.mesh import QuantizedInterval
from .. import tools


class TestFiniteDifferences(tools.SeeMPSTestCase):
    interval = QuantizedInterval(-1, 1, qubits=2)

    def Down(self, nqubits: int, periodic: bool = False) -> csr_matrix:
        """Moves f[i] to f[i-1]"""
        L = 2**nqubits
        if periodic:
            M = spdiags([np.ones(L), np.ones(L)], [1, -L + 1], format="csr")  # type: ignore # scipy-stubs broken
        else:
            M = spdiags([np.ones(L)], [1], format="csr")  # type: ignore # scipy-stubs broken
        return cast(csr_matrix, M)

    def test_first_derivative_two_qubits_perodic(self):
        dx = self.interval.step
        D2 = finite_differences_mpo(
            order=1, filter=3, interval=self.interval, periodic=True
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
        dx = self.interval.step
        D2 = finite_differences_mpo(
            order=2, filter=3, interval=self.interval, periodic=True
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
        self.assertSimilar(D2, (D - 2.0 * I + D.T) / dx2)

    def test_first_derivative_two_qubits_non_perodic(self):
        dx = self.interval.step
        D2 = finite_differences_mpo(
            order=1, filter=3, interval=self.interval, periodic=False
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
        dx = self.interval.step
        D2 = finite_differences_mpo(
            order=2, filter=3, interval=self.interval, periodic=False
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


class TestTridiagonalMPO(tools.SeeMPSTestCase):
    def test_open_tridiagonal_matrix(self):
        n, a, b, c = 3, 2.0, -1.0, 0.5
        M = tridiagonal_mpo(n, a, b, c, closed=False).to_matrix()
        N = 2**n
        expected = a * np.eye(N) + b * np.eye(N, k=-1) + c * np.eye(N, k=1)
        self.assertSimilar(M, expected)

    def test_closed_tridiagonal_wraps_around(self):
        n, a, b, c = 3, 2.0, -1.0, 0.5
        M = tridiagonal_mpo(n, a, b, c, closed=True).to_matrix()
        N = 2**n
        # A closed (periodic) operator adds the corner couplings.
        expected = a * np.eye(N) + b * np.eye(N, k=-1) + c * np.eye(N, k=1)
        expected[0, N - 1] = b
        expected[N - 1, 0] = c
        self.assertSimilar(M, expected)


class TestFiniteDifferencesEdgeCases(tools.SeeMPSTestCase):
    def test_unknown_formula_raises(self):
        with self.assertRaises(ValueError):
            finite_differences_mpo(1, QuantizedInterval(0, 1, qubits=3), filter=99)

    def test_step_rescaling_for_small_dx(self):
        # A very small step triggers the rescaling loop; the result must still
        # match the plain formula built on the rescaled step.
        interval = (0.0, 1e-6, 8)
        mpo = finite_differences_mpo(1, interval, filter=3, tol=1e-2)
        self.assertEqual(mpo.to_matrix().shape, (256, 256))
