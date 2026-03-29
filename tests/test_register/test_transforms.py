import numpy as np
import scipy.sparse as sp  # type: ignore
from seemps.analysis.interpolation import twos_complement
from seemps.state import MPS
from seemps.register import qubo_exponential_mpo, qubo_mpo, twoscomplement
from seemps.operators import MPO, MPOList
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

    def test_qubo_exponential_for_magnetic_field_returns_diagonal_mpo(self):
        h = np.array([0.25, -0.5, 1.0])
        op0 = qubo_exponential_mpo(h=h, beta=0.0)
        self.assertIsInstance(op0, MPO)
        self.assertSimilar(op0.to_matrix(), np.eye(2 ** len(h)))

        beta1 = -0.7
        beta2 = 0.2
        op1 = qubo_exponential_mpo(h=h, beta=beta1).to_matrix()
        op2 = qubo_exponential_mpo(h=h, beta=beta2).to_matrix()
        op12 = qubo_exponential_mpo(h=h, beta=beta1 + beta2).to_matrix()
        self.assertSimilar(op1, np.diag(np.diag(op1)))
        self.assertSimilar(op1 @ op2, op12)

    def test_qubo_exponential_for_interactions_returns_diagonal_mpolist(self):
        J = np.array([[0.1, 0.2, -0.1], [0.0, -0.3, 0.4], [0.5, 0.1, 0.2]])
        h = np.array([0.3, -0.2, 0.1])
        op0 = qubo_exponential_mpo(J=J, h=h, beta=0.0)
        self.assertIsInstance(op0, MPOList)
        self.assertSimilar(op0.to_matrix(), np.eye(2 ** len(h)))

        beta1 = 0.4
        beta2 = -0.15
        op1 = qubo_exponential_mpo(J=J, h=h, beta=beta1)
        op2 = qubo_exponential_mpo(J=J, h=h, beta=beta2)
        op12 = qubo_exponential_mpo(J=J, h=h, beta=beta1 + beta2)
        self.assertTrue(all(mpo.size == len(h) for mpo in op1.mpos))
        self.assertSimilar(op1.to_matrix(), np.diag(np.diag(op1.to_matrix())))
        self.assertSimilar(op1.to_matrix() @ op2.to_matrix(), op12.to_matrix())

    def test_qubo_exponential_small_one_site_problem(self):
        h = np.array([0.5])
        beta = 2.0
        expected = np.diag([1.0, np.exp(beta * 0.5)])
        self.assertSimilar(qubo_exponential_mpo(h=h, beta=beta).to_matrix(), expected)

    def test_qubo_exponential_small_two_site_asymmetric_problem(self):
        J = np.array([[0.25, 0.5], [-0.2, -0.4]])
        h = np.array([0.1, -0.3])
        beta = -1.5
        # Energies for |00>, |01>, |10>, |11> are 0, -0.7, 0.35, -0.05.
        expected = np.diag(
            [
                1.0,
                np.exp(-1.5 * -0.7),
                np.exp(-1.5 * 0.35),
                np.exp(-1.5 * -0.05),
            ]
        )
        self.assertSimilar(qubo_exponential_mpo(J=J, h=h, beta=beta).to_matrix(), expected)

    def test_qubo_exponential_small_three_site_problem(self):
        J = np.array([[0.2, 0.4, 0.0], [0.1, -0.3, -0.2], [0.0, 0.5, 0.1]])
        h = np.array([0.0, 0.2, -0.4])
        beta = 0.5
        # Energies for |000> ... |111> are
        # 0, -0.3, -0.1, -0.1, 0.2, -0.1, 0.6, 0.6.
        expected = np.diag(
            [
                1.0,
                np.exp(-0.15),
                np.exp(-0.05),
                np.exp(-0.05),
                np.exp(0.1),
                np.exp(-0.05),
                np.exp(0.3),
                np.exp(0.3),
            ]
        )
        self.assertSimilar(qubo_exponential_mpo(J=J, h=h, beta=beta).to_matrix(), expected)

    def test_twoscomplement_matches_known_three_qubit_action(self):
        expected = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0],
            ],
            dtype=float,
        )
        self.assertSimilar(twoscomplement(3).to_matrix(), expected)
        self.assertSimilar(twoscomplement(3).to_matrix(), twos_complement(3).to_matrix())

    def test_twoscomplement_sites_branch_extends_selected_register(self):
        sites = [0, 2, 4]
        extended = twoscomplement(5, control=2, sites=sites)
        reduced = twoscomplement(len(sites), control=sites.index(2)).extend(5, sites=sites)
        self.assertSimilar(extended.to_matrix(), reduced.to_matrix())
