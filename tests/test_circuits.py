import numpy as np
from .tools import TestCase
from seemps import random_mps, CanonicalMPS
from seemps.register.circuit import (
    LocalRotationsLayer,
    TwoQubitGatesLayer,
    VQECircuit,
    interpret_operator,
)
from scipy.linalg import expm  # type: ignore


class TestLocalGateCircuits(TestCase):
    Sx = interpret_operator("Sx")
    Sy = interpret_operator("Sy")
    Sz = interpret_operator("Sz")

    def test_local_gates_matching_parameters_and_register_size(self):
        """When not `same_parameter`, number of parameters must match number
        of register qubits. Otherwise they are just one default parameter."""
        with self.assertRaises(Exception):
            LocalRotationsLayer(3, "Sz", same_parameter=False, default_parameters=[0.0])
        with self.assertRaises(Exception):
            LocalRotationsLayer(
                3, "Sz", same_parameter=True, default_parameters=[0.0] * 3
            )

    def test_local_sx_gate_one_qubit(self):
        a = self.random_mps(2, 1, truncate=True, normalize=True)
        U = LocalRotationsLayer(1, "Sx", default_parameters=[0.35])
        self.assertSimilar(
            U.apply(a).to_vector(), expm(-1j * 0.35 * self.Sx) @ a.to_vector()
        )

    def test_local_sx_gate_two_qubits_same_angle(self):
        a = self.random_mps(2, 2, truncate=True, normalize=True)
        U = LocalRotationsLayer(2, "Sx", same_parameter=True, default_parameters=[0.35])
        U1 = expm(-1j * 0.35 * self.Sx)
        U2 = expm(-1j * 0.35 * self.Sx)
        self.assertSimilar(U.apply(a).to_vector(), np.kron(U1, U2) @ a.to_vector())

    def test_local_sx_gate_two_qubits_different_angles(self):
        a = self.random_mps(2, 2, truncate=True, normalize=True)
        U = LocalRotationsLayer(2, "Sx", default_parameters=[0.35, 0.13])
        U1 = expm(-1j * 0.35 * self.Sx)
        U2 = expm(-1j * 0.13 * self.Sx)
        self.assertSimilar(U.apply(a).to_vector(), np.kron(U1, U2) @ a.to_vector())

    def test_local_sy_gate_one_qubit(self):
        a = self.random_mps(2, 1, truncate=True, normalize=True)
        U = LocalRotationsLayer(1, "Sy", default_parameters=[0.35])
        self.assertSimilar(
            U.apply(a).to_vector(), expm(-1j * 0.35 * self.Sy) @ a.to_vector()
        )

    def test_local_sy_gate_two_qubits_same_angle(self):
        a = self.random_mps(2, 2, truncate=True, normalize=True)
        U = LocalRotationsLayer(2, "Sy", same_parameter=True, default_parameters=[0.35])
        U1 = expm(-1j * 0.35 * self.Sy)
        U2 = expm(-1j * 0.35 * self.Sy)
        self.assertSimilar(U.apply(a).to_vector(), np.kron(U1, U2) @ a.to_vector())

    def test_local_sy_gate_two_qubits_different_angles(self):
        a = self.random_mps(2, 2, truncate=True, normalize=True)
        U = LocalRotationsLayer(2, "Sy", default_parameters=[0.35, 0.13])
        U1 = expm(-1j * 0.35 * self.Sy)
        U2 = expm(-1j * 0.13 * self.Sy)
        self.assertSimilar(U.apply(a).to_vector(), np.kron(U1, U2) @ a.to_vector())

    def test_local_sz_gate_one_qubit(self):
        a = self.random_mps(2, 1, truncate=True, normalize=True)
        U = LocalRotationsLayer(1, "Sz", default_parameters=[0.35])
        self.assertSimilar(
            U.apply(a).to_vector(), expm(-1j * 0.35 * self.Sz) @ a.to_vector()
        )

    def test_local_sz_gate_two_qubits_same_angle(self):
        a = self.random_mps(2, 2, truncate=True, normalize=True)
        U = LocalRotationsLayer(2, "Sz", same_parameter=True, default_parameters=[0.35])
        U1 = expm(-1j * 0.35 * self.Sz)
        U2 = expm(-1j * 0.35 * self.Sz)
        self.assertSimilar(U.apply(a).to_vector(), np.kron(U1, U2) @ a.to_vector())

    def test_local_sz_gate_two_qubits_different_angles(self):
        a = self.random_mps(2, 2, truncate=True, normalize=True)
        U = LocalRotationsLayer(2, "Sz", default_parameters=[0.35, 0.13])
        U1 = expm(-1j * 0.35 * self.Sz)
        U2 = expm(-1j * 0.13 * self.Sz)
        self.assertSimilar(U.apply(a).to_vector(), np.kron(U1, U2) @ a.to_vector())


class TestEntanglingLayerCircuit(TestCase):
    CNOT = interpret_operator("CNOT")
    CZ = interpret_operator("CZ")

    def test_single_cnot_circuit(self):
        a = self.random_mps(2, 2, truncate=True, normalize=True)
        U = TwoQubitGatesLayer(2, "CNOT")
        self.assertSimilar(U.apply(a).to_vector(), self.CNOT @ a.to_vector())

    def test_two_cnot_circuit(self):
        a = self.random_mps(2, 3, truncate=True, normalize=True)
        U = TwoQubitGatesLayer(3, "CNOT")
        U1 = np.kron(self.CNOT, np.eye(2))
        U2 = np.kron(np.eye(2), self.CNOT)
        self.assertSimilar(U.apply(a).to_vector(), U2 @ U1 @ a.to_vector())

    def test_three_cnot_circuit(self):
        a = self.random_mps(2, 4, truncate=True, normalize=True)
        U = TwoQubitGatesLayer(4, "CNOT")
        U1 = np.kron(self.CNOT, np.eye(4))
        U2 = np.kron(np.kron(np.eye(2), self.CNOT), np.eye(2))
        U3 = np.kron(np.eye(4), self.CNOT)
        self.assertSimilar(U.apply(a).to_vector(), U3 @ U2 @ U1 @ a.to_vector())

    def test_single_cz_circuit(self):
        a = self.random_mps(2, 2, truncate=True, normalize=True)
        U = TwoQubitGatesLayer(2, "CZ")
        self.assertSimilar(U.apply(a).to_vector(), self.CZ @ a.to_vector())

    def test_two_cz_circuit(self):
        a = self.random_mps(2, 3, truncate=True, normalize=True)
        U = TwoQubitGatesLayer(3, "CZ")
        U1 = np.kron(self.CZ, np.eye(2))
        U2 = np.kron(np.eye(2), self.CZ)
        self.assertSimilar(U.apply(a).to_vector(), U2 @ U1 @ a.to_vector())

    def test_three_cz_circuit(self):
        a = self.random_mps(2, 4, truncate=True, normalize=True)
        U = TwoQubitGatesLayer(4, "CZ")
        U1 = np.kron(self.CZ, np.eye(4))
        U2 = np.kron(np.kron(np.eye(2), self.CZ), np.eye(2))
        U3 = np.kron(np.eye(4), self.CZ)
        self.assertSimilar(U.apply(a).to_vector(), U3 @ U2 @ U1 @ a.to_vector())


class TestVQE(TestCase):
    Sy = interpret_operator("Sy")
    CNOT = interpret_operator("CNOT")

    def test_VQE_two_qubits_one_layer(self):
        a = self.random_mps(2, 2, truncate=True, normalize=True)
        U = VQECircuit(2, 1, 2 * [0.13])
        U1 = np.kron(expm(-1j * 0.13 * self.Sy), expm(-1j * 0.13 * self.Sy))
        U2 = self.CNOT
        self.assertSimilar(U.apply(a).to_vector(), U2 @ U1 @ a.to_vector())

    def test_VQE_two_qubits_two_layers(self):
        a = self.random_mps(2, 2, truncate=True, normalize=True)
        U = VQECircuit(2, 2, [0.13, -0.5, -0.25, 0.73])
        U1 = np.kron(expm(-1j * 0.13 * self.Sy), expm(-1j * -0.5 * self.Sy))
        U2 = self.CNOT
        U3 = np.kron(expm(-1j * -0.25 * self.Sy), expm(-1j * 0.73 * self.Sy))
        U4 = self.CNOT
        self.assertSimilar(U.apply(a).to_vector(), U4 @ U3 @ U2 @ U1 @ a.to_vector())

    def test_VQE_entangling_layer_order(self):
        a = self.random_mps(2, 3, truncate=True, normalize=True)
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
        a = self.random_mps(2, 2, truncate=True, normalize=True)
        U = VQECircuit(2, 1, 2 * [0.0])
        U1 = np.kron(expm(-1j * 0.13 * self.Sy), expm(-1j * 0.15 * self.Sy))
        U2 = self.CNOT
        self.assertSimilar(
            U.apply(a, [0.13, 0.15]).to_vector(), U2 @ U1 @ a.to_vector()
        )
