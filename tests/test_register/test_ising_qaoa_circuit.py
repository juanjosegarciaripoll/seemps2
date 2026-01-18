import numpy as np
from scipy.linalg import expm
from ..tools import TestCase
from seemps.register import IsingQAOACircuit, interpret_operator, qubo_mpo


class TestIsingQAOACircuit(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.J = np.asarray([[0, 1, 0], [1, 0, 2], [0, 2, 0]])
        self.nqubits = len(self.J)
        self.H = qubo_mpo(self.J, None)
        Hadamard = interpret_operator("H")
        self.Hadamards = np.kron(Hadamard, np.kron(Hadamard, Hadamard))

        sy = interpret_operator("σY")
        id2 = np.eye(2)
        self.Hlocal = (
            np.kron(sy, np.kron(id2, id2))
            + np.kron(id2, np.kron(sy, id2))
            + np.kron(id2, np.kron(id2, sy))
        )
        self.Hmatrix = self.H.to_matrix()

    def test_ising_qaoa_no_layers_is_hadamards(self) -> None:
        a = self.random_uniform_mps(2, self.H.size, truncate=True, normalize=True)
        U = IsingQAOACircuit(self.J, None, layers=0)
        Ustate = U.apply(a)
        self.assertSimilar(Ustate.to_vector(), self.Hadamards @ a.to_vector())

    def test_ising_qaoa_one_layer_zero_parameters(self) -> None:
        a = self.random_uniform_mps(2, self.H.size, truncate=True, normalize=True)
        U = IsingQAOACircuit(self.J, None, layers=1)
        parameters = [0.0, 0.0]
        Ustate = U.apply(a, parameters)

        self.assertSimilar(
            Ustate.to_vector(),
            self.Hadamards @ a.to_vector(),
        )

    def test_ising_qaoa_one_layer_zero_Hamiltonian(self) -> None:
        a = self.random_uniform_mps(2, self.H.size, truncate=True, normalize=True)
        U = IsingQAOACircuit(self.J, None, layers=1)
        parameters = [0.0, 1.0]
        Ustate = U.apply(a.copy(), parameters)

        self.assertSimilar(
            Ustate.to_vector(),
            expm(-1j * parameters[1] * self.Hlocal) @ self.Hadamards @ a.to_vector(),
        )

    def test_ising_qaoa_one_layer_zero_local(self) -> None:
        a = self.random_uniform_mps(2, self.H.size, truncate=True, normalize=True)
        U = IsingQAOACircuit(self.J, None, layers=1)
        parameters = [1.0, 0.0]
        Ustate = U.apply(a.copy(), parameters)

        self.assertSimilar(
            Ustate.to_vector(),
            expm(-1j * parameters[0] * self.Hmatrix) @ self.Hadamards @ a.to_vector(),
        )

    def test_ising_qaoa_one_layer(self) -> None:
        a = self.random_uniform_mps(2, self.H.size, truncate=True, normalize=True)
        U = IsingQAOACircuit(self.J, None, layers=1)
        parameters = [1.0, -np.pi]
        Ustate = U.apply(a.copy(), parameters)

        self.assertSimilar(
            Ustate.to_vector(),
            expm(-1j * parameters[1] * self.Hlocal)
            @ expm(-1j * parameters[0] * self.Hmatrix)
            @ self.Hadamards
            @ a.to_vector(),
        )

    def test_ising_qaoa_two_layers(self) -> None:
        a = self.random_uniform_mps(2, self.H.size, truncate=True, normalize=True)
        U = IsingQAOACircuit(self.J, None, layers=2)
        parameters = [1.0, -np.pi, 0.23, 0.1 * np.pi]
        Ustate = U.apply(a.copy(), parameters)

        self.assertSimilar(
            Ustate.to_vector(),
            expm(-1j * parameters[3] * self.Hlocal)
            @ expm(-1j * parameters[2] * self.Hmatrix)
            @ expm(-1j * parameters[1] * self.Hlocal)
            @ expm(-1j * parameters[0] * self.Hmatrix)
            @ self.Hadamards
            @ a.to_vector(),
        )
