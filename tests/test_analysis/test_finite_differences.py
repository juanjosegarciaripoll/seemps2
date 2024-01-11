import numpy as np
from seemps.analysis.finite_differences import *
from seemps.analysis.space import Space
from seemps.state import MPS
from seemps.state.core import DEFAULT_STRATEGY, NO_TRUNCATION

from .tools import *


def gaussian(x):
    f = np.exp(-(x**2))
    return f / np.linalg.norm(f)


def S_plus_v(n, closed=True):
    S = np.diag(np.ones(2**n - 1), +1)
    if closed:
        S[-1, 0] = 1
    return S


def S_minus_v(n, closed=True):
    S = np.diag(np.ones(2**n - 1), -1)
    if closed:
        S[0, -1] = 1
    return S


def finite_differences_v(n, Δx, closed=True):
    return (1 / Δx**2) * (
        S_plus_v(n, closed=closed) + S_minus_v(n, closed=closed) - 2 * np.eye(2**n)
    )


class TestFiniteDifferences(TestCase):
    def test_finite_differences(self):
        for n in range(2, 10):
            qubits_per_dimension = [n]
            dims = [2**n for n in qubits_per_dimension]
            L = 10
            space = Space(qubits_per_dimension, L=[[-L / 2, L / 2]])
            x = space.x[0]
            Δx = space.dx[0]
            v = gaussian(x)
            fd_sol = finite_differences_v(n, Δx, closed=True) @ v
            ψ = MPS.from_vector(v, [2] * n, normalize=False, strategy=NO_TRUNCATION)
            fd_mps_sol = (
                finite_differences_mpo(n, Δx, closed=True, strategy=NO_TRUNCATION) @ ψ
            )
            self.assertSimilar(fd_sol, fd_mps_sol)
