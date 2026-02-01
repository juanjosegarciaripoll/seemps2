import numpy as np
from ..tools import SeeMPSTestCase
from seemps.register.circuit import VQECircuit, interpret_operator
from scipy.linalg import expm


class TestVQE(SeeMPSTestCase):
    Sy = interpret_operator("Sy")
    CNOT = interpret_operator("CNOT")

    def test_VQE_two_qubits_one_layer(self):
        a = self.random_uniform_canonical_mps(2, 2, truncate=True, normalize=True)
        U = VQECircuit(2, 1, 2 * [0.13])
        U1 = np.kron(expm(-1j * 0.13 * self.Sy), expm(-1j * 0.13 * self.Sy))
        U2 = self.CNOT
        self.assertSimilar(U.apply(a).to_vector(), U2 @ U1 @ a.to_vector())

    def test_VQE_two_qubits_two_layers(self):
        a = self.random_uniform_canonical_mps(2, 2, truncate=True, normalize=True)
        U = VQECircuit(2, 2, [0.13, -0.5, -0.25, 0.73])
        U1 = np.kron(expm(-1j * 0.13 * self.Sy), expm(-1j * -0.5 * self.Sy))
        U2 = self.CNOT
        U3 = np.kron(expm(-1j * -0.25 * self.Sy), expm(-1j * 0.73 * self.Sy))
        U4 = self.CNOT
        self.assertSimilar(U.apply(a).to_vector(), U4 @ U3 @ U2 @ U1 @ a.to_vector())

    def test_VQE_entangling_layer_order(self):
        a = self.random_uniform_canonical_mps(2, 3, truncate=True, normalize=True)
        U = VQECircuit(3, 2, [0.13] * 3 * 2)
        U1 = expm(-1j * 0.13 * self.Sy)
        Ulocal = np.kron(U1, np.kron(U1, U1))
        Uright_to_left = np.kron(self.CNOT, np.eye(2)) @ np.kron(np.eye(2), self.CNOT)
        Uleft_to_right = np.kron(np.eye(2), self.CNOT) @ np.kron(self.CNOT, np.eye(2))
        self.assertSimilar(
            U.apply(a).to_vector(),
            Uright_to_left @ Ulocal @ Uleft_to_right @ Ulocal @ a.to_vector(),
        )

    def test_VQE_apply_uses_parameters(self):
        a = self.random_uniform_canonical_mps(2, 2, truncate=True, normalize=True)
        U = VQECircuit(2, 1, 2 * [0.0])
        U1 = np.kron(expm(-1j * 0.13 * self.Sy), expm(-1j * 0.15 * self.Sy))
        U2 = self.CNOT
        self.assertSimilar(
            U.apply(a, [0.13, 0.15]).to_vector(), U2 @ U1 @ a.to_vector()
        )
