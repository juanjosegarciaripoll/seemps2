import numpy as np
from scipy.linalg import expm
from seemps.operators import MPO
from seemps.hamiltonians import HeisenbergHamiltonian, ConstantTIHamiltonian
from seemps.register.circuit import HamiltonianEvolutionLayer, interpret_operator
from ..tools import TestCase


class TestHamiltonianEvolutionLayer(TestCase):
    H: MPO

    def setUp(self) -> None:
        super().setUp()
        self.H = HeisenbergHamiltonian(4).to_mpo()
        self.Hzero = MPO.from_local_operators([np.zeros((2, 2))] * 4)

    def test_hamiltonian_layer_accepts_mpo(self):
        L = HamiltonianEvolutionLayer(self.H)
        self.assertIsInstance(L, HamiltonianEvolutionLayer)

    def test_hamiltonian_layer_default_order(self):
        L = HamiltonianEvolutionLayer(self.H)
        self.assertEqual(L.order, 6)

    def test_hamiltonian_layer_requires_qubit_operators(self):
        SX = interpret_operator("SX(1)")
        H = ConstantTIHamiltonian(4, np.kron(SX, SX)).to_mpo()
        with self.assertRaises(Exception):
            HamiltonianEvolutionLayer(H)

    def test_zero_hamiltonian_layer_equal_to_exponential(self):
        a = self.random_uniform_mps(2, self.Hzero.size, truncate=True, normalize=True)
        U = HamiltonianEvolutionLayer(self.Hzero)
        c = U.apply_inplace(a.copy())
        self.assertSimilar(
            c.to_vector(),
            a.to_vector(),
        )

    def test_hamiltonian_layer_equal_to_exponential(self):
        a = self.random_uniform_mps(2, self.H.size, truncate=True, normalize=True)
        U = HamiltonianEvolutionLayer(self.H)
        c = U.apply_inplace(a.copy())
        self.assertSimilar(
            expm(-1j * U.parameters[0] * self.H.to_matrix()) @ c.to_vector(),
            a.to_vector(),
        )

    def test_hamiltonian_layer_users_parameters(self):
        a = self.random_uniform_mps(2, self.H.size, truncate=True, normalize=True)
        U = HamiltonianEvolutionLayer(self.H)
        g = 0.23
        c = U.apply_inplace(a.copy(), parameters=[g])
        self.assertSimilar(
            expm(-1j * g * self.H.to_matrix()) @ c.to_vector(),
            a.to_vector(),
        )
