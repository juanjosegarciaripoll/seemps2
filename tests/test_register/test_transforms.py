import numpy as np
import scipy.sparse as sp  # type: ignore
from seemps.state import MPS
from seemps.register import qubo_mpo
from ..tools import TestCase


class TestAlgebraic(TestCase):
    P1 = sp.diags([0.0, 1.0], 0)
    i2 = sp.eye(2, dtype=np.float64)

    @classmethod
    def projector(cls, i, L):
        return sp.kron(sp.eye(2**i), sp.kron(cls.P1, sp.eye(2 ** (L - i - 1))))

    @classmethod
    def linear_operator(cls, h):
        L = len(h)
        return sum(hi * cls.projector(i, L) for i, hi in enumerate(h) if hi)

    @classmethod
    def quadratic_operator(cls, J):
        L = len(J)
        return sum(
            J[i, j] * (cls.projector(i, L) @ cls.projector(j, L))
            for i in range(L)
            for j in range(L)
            if J[i, j]
        )

    def test_qubo_magnetic_field(self):
        np.random.seed(1022)
        for N in range(1, 10):
            h = np.random.rand(N) - 0.5
            self.assertSimilar(qubo_mpo(h=h).to_matrix(), self.linear_operator(h))

    def test_qubo_quadratic(self):
        np.random.seed(1022)
        for N in range(1, 10):
            J = np.random.rand(N, N) - 0.5
            self.assertSimilar(qubo_mpo(J=J).to_matrix(), self.quadratic_operator(J))

    def test_product(self):
        np.random.seed(1034)
        for N in range(1, 10):
            ψ = np.random.rand(2**N, 2) - 0.5
            ψ = ψ[:, 0] + 1j * ψ[:, 1]
            ψ /= np.linalg.norm(ψ)
            ψmps = MPS.from_vector(ψ, [2] * N)
            ψ = ψmps.to_vector()

            ξ = np.random.rand(2**N, 2) - 0.5
            ξ = ξ[:, 0] + 1j * ξ[:, 1]
            ξ /= np.linalg.norm(ξ)
            ξmps = MPS.from_vector(ξ, [2] * N)
            ξ = ξmps.to_vector()

            self.assertSimilar(ψmps * ξmps, ψ * ξ)
