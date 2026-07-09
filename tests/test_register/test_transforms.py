import numpy as np
import scipy.sparse as sp  # type: ignore
from seemps.state import MPS
from seemps.register import qubo_mpo
from seemps.register.transforms import mpo_shifts, twoscomplement
from ..tools import SeeMPSTestCase


class TestAlgebraic(SeeMPSTestCase):
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
        for N in range(1, 10):
            h = self.rng.random(size=N) - 0.5
            self.assertSimilar(qubo_mpo(h=h).to_matrix(), self.linear_operator(h))

    def test_qubo_quadratic(self):
        for N in range(1, 10):
            J = self.rng.random(size=(N, N)) - 0.5
            self.assertSimilar(qubo_mpo(J=J).to_matrix(), self.quadratic_operator(J))

    def test_product(self):
        for N in range(1, 10):
            ψ = self.rng.random(size=(2**N, 2)) - 0.5
            ψ = ψ[:, 0] + 1j * ψ[:, 1]
            ψ /= np.linalg.norm(ψ)
            ψmps = MPS.from_vector(ψ, [2] * N)
            ψ = ψmps.to_vector()

            ξ = self.rng.random(size=(2**N, 2)) - 0.5
            ξ = ξ[:, 0] + 1j * ξ[:, 1]
            ξ /= np.linalg.norm(ξ)
            ξmps = MPS.from_vector(ξ, [2] * N)
            ξ = ξmps.to_vector()

            self.assertSimilar(ψmps * ξmps, ψ * ξ)

    def test_qubo_combined_field_and_coupling(self):
        # Providing both J and h folds h into the diagonal of J (qubo.py:53).
        for N in range(1, 6):
            J = self.rng.random(size=(N, N)) - 0.5
            h = self.rng.random(size=N) - 0.5
            expected = self.quadratic_operator(J) + self.linear_operator(h)
            self.assertSimilar(qubo_mpo(J=J, h=h).to_matrix(), expected)

    def test_qubo_requires_some_argument(self):
        with self.assertRaises(ValueError):
            qubo_mpo()


class TestShiftsAndComplement(SeeMPSTestCase):
    def test_mpo_shifts_tuple_matches_list(self):
        # A tuple is interpreted as a range of displacements.
        for L in range(1, 5):
            from_tuple = mpo_shifts(L, (0, 3)).to_matrix()
            from_list = mpo_shifts(L, [0, 1, 2]).to_matrix()
            self.assertSimilar(from_tuple, from_list)

    def test_twoscomplement_is_an_involution(self):
        # Two's complement negates the register, so applying it twice is the
        # identity; the operator is also a 0/1 permutation matrix.
        for L in range(1, 5):
            O = twoscomplement(L).to_matrix()
            self.assertSimilar(O @ O, np.eye(2**L))
            self.assertSimilar(O @ O.T, np.eye(2**L))

    def test_twoscomplement_on_a_subset_of_sites(self):
        L = 4
        O = twoscomplement(L, control=1, sites=[1, 2, 3]).to_matrix()
        self.assertEqual(O.shape, (2**L, 2**L))
        self.assertSimilar(O @ O, np.eye(2**L))
