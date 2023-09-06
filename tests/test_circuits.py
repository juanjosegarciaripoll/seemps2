import numpy as np
from .tools import TestCase
from seemps import random_mps, CanonicalMPS
from seemps.register.circuit import (
    LocalRotationsLayer,
    TwoQubitGatesLayer,
    interpret_operator,
)
from scipy.linalg import expm


class TestLocalGateCircuits(TestCase):
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

    def test_local_sz_gate_one_qubit(self):
        a = CanonicalMPS(random_mps(2, 1, truncate=True), normalize=True)
        U = LocalRotationsLayer(1, "Sz", default_parameters=[0.35])
        self.assertSimilar(
            U.apply(a).to_vector(), expm(-1j * 0.35 * self.Sz) @ a.to_vector()
        )

    def test_local_sz_gate_two_qubits_same_angle(self):
        a = CanonicalMPS(random_mps(2, 2, truncate=True), normalize=True)
        U = LocalRotationsLayer(2, "Sz", same_parameter=True, default_parameters=[0.35])
        U1 = expm(-1j * 0.35 * self.Sz)
        U2 = expm(-1j * 0.35 * self.Sz)
        self.assertSimilar(U.apply(a).to_vector(), np.kron(U1, U2) @ a.to_vector())

    def test_local_sz_gate_two_qubits_different_angles(self):
        a = CanonicalMPS(random_mps(2, 2, truncate=True), normalize=True)
        U = LocalRotationsLayer(2, "Sz", default_parameters=[0.35, 0.13])
        U1 = expm(-1j * 0.35 * self.Sz)
        U2 = expm(-1j * 0.13 * self.Sz)
        self.assertSimilar(U.apply(a).to_vector(), np.kron(U1, U2) @ a.to_vector())


class TestEntanglingLayerCircuit(TestCase):
    CNOT = interpret_operator("CNOT")
    CZ = interpret_operator("CZ")

    def test_single_cnot_circuit(self):
        a = CanonicalMPS(random_mps(2, 2, truncate=True), normalize=True)
        U = TwoQubitGatesLayer(2, "CNOT")
        self.assertSimilar(U.apply(a).to_vector(), self.CNOT @ a.to_vector())

    def test_two_cnot_circuit(self):
        a = CanonicalMPS(random_mps(2, 3, truncate=True), normalize=True)
        U = TwoQubitGatesLayer(3, "CNOT")
        U1 = np.kron(self.CNOT, np.eye(2))
        U2 = np.kron(np.eye(2), self.CNOT)
        self.assertSimilar(U.apply(a).to_vector(), U2 @ U1 @ a.to_vector())

    def test_three_cnot_circuit(self):
        a = CanonicalMPS(random_mps(2, 3, truncate=True), normalize=True)
        U = TwoQubitGatesLayer(3, "CNOT")
        U1 = np.kron(self.CNOT, np.eye(2))
        U2 = np.kron(np.eye(2), self.CNOT)
        self.assertSimilar(U.apply(a).to_vector(), U2 @ U1 @ a.to_vector())

    def test_single_cz_circuit(self):
        a = CanonicalMPS(random_mps(2, 2, truncate=True), normalize=True)
        U = TwoQubitGatesLayer(2, "CZ")
        self.assertSimilar(U.apply(a).to_vector(), self.CZ @ a.to_vector())

    def test_two_cz_circuit(self):
        a = CanonicalMPS(random_mps(2, 3, truncate=True), normalize=True)
        U = TwoQubitGatesLayer(3, "CZ")
        U1 = np.kron(self.CZ, np.eye(2))
        U2 = np.kron(np.eye(2), self.CZ)
        self.assertSimilar(U.apply(a).to_vector(), U2 @ U1 @ a.to_vector())

    def test_three_cz_circuit(self):
        a = CanonicalMPS(random_mps(2, 3, truncate=True), normalize=True)
        U = TwoQubitGatesLayer(3, "CZ")
        U1 = np.kron(self.CZ, np.eye(2))
        U2 = np.kron(np.eye(2), self.CZ)
        self.assertSimilar(U.apply(a).to_vector(), U2 @ U1 @ a.to_vector())
