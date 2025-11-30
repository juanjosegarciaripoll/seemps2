import numpy as np

from seemps.state.mps import MPS
from .tools import TestCase
from seemps.state import CanonicalMPS
from seemps.register.circuit import (
    LocalRotationsLayer,
    TwoQubitGatesLayer,
    VQECircuit,
    interpret_operator,
)
from scipy.linalg import expm  # type: ignore


class TestKnownOperators(TestCase):
    def test_interpret_operator_is_case_insensitive(self):
        self.assertSimilar(interpret_operator("Sx"), interpret_operator("SX"))

    def test_interpret_operator_accepts_aliases(self):
        self.assertSimilar(interpret_operator("CX"), interpret_operator("CNOT"))

    def test_interpret_operator_signals_unknown_operators(self):
        with self.assertRaises(Exception):
            interpret_operator("CUp")

    def test_interpret_operator_only_accepts_strings_or_matrices(self):
        interpret_operator("Sz")
        interpret_operator(np.eye(2))
        with self.assertRaises(Exception):
            interpret_operator(1)  # type: ignore
        with self.assertRaises(Exception):
            interpret_operator(np.zeros((3, 1)))  # type: ignore
        with self.assertRaises(Exception):
            interpret_operator(np.zeros((3,)))  # type: ignore


class TestLocalGateCircuits(TestCase):
    Sx = interpret_operator("Sx")
    Sy = interpret_operator("Sy")
    Sz = interpret_operator("Sz")

    def test_local_gates_matching_parameters_and_register_size(self):
        """When not `same_parameter`, number of parameters must match number
        of register qubits. Otherwise they are just one default parameter."""
        with self.assertRaises(Exception):
            LocalRotationsLayer(
                3,
                "Sz",
                same_parameter=False,
                default_parameters=[0.0],  # type: ignore
            )
        with self.assertRaises(Exception):
            LocalRotationsLayer(
                3,
                "Sz",
                same_parameter=True,
                default_parameters=[0.0] * 3,  # type: ignore
            )

    def test_local_gates_requires_qubit_operators(self):
        with self.assertRaises(Exception):
            LocalRotationsLayer(3, "CNOT")

    def test_local_gates_apply_in_place_only_modifies_canonical_mps(self):
        a = self.random_uniform_mps(2, 1, truncate=True, normalize=True)
        U = LocalRotationsLayer(1, "Sx", default_parameters=[0.35])
        b = U.apply_inplace(a)
        self.assertTrue(a is b)
        a = MPS(a)
        b = U.apply_inplace(a)
        self.assertFalse(b is a)

    def test_local_gates_matmul_works(self):
        a = self.random_uniform_mps(2, 1, truncate=True, normalize=True)
        U = LocalRotationsLayer(1, "Sx", default_parameters=[0.35])
        b = a.copy()
        c = U.apply_inplace(a)
        d = U @ b
        self.assertSimilar(c, d)

    def test_local_sx_gate_one_qubit(self):
        a = self.random_uniform_mps(2, 1, truncate=True, normalize=True)
        U = LocalRotationsLayer(1, "Sx", default_parameters=[0.35])
        self.assertSimilar(
            U.apply(a).to_vector(), expm(-1j * 0.35 * self.Sx) @ a.to_vector()
        )

    def test_local_sx_gate_two_qubits_same_angle(self):
        a = self.random_uniform_mps(2, 2, truncate=True, normalize=True)
        U = LocalRotationsLayer(2, "Sx", same_parameter=True, default_parameters=[0.35])
        U1 = expm(-1j * 0.35 * self.Sx)
        U2 = expm(-1j * 0.35 * self.Sx)
        self.assertSimilar(U.apply(a).to_vector(), np.kron(U1, U2) @ a.to_vector())

    def test_local_sx_gate_two_qubits_different_angles(self):
        a = self.random_uniform_mps(2, 2, truncate=True, normalize=True)
        U = LocalRotationsLayer(2, "Sx", default_parameters=[0.35, 0.13])
        U1 = expm(-1j * 0.35 * self.Sx)
        U2 = expm(-1j * 0.13 * self.Sx)
        self.assertSimilar(U.apply(a).to_vector(), np.kron(U1, U2) @ a.to_vector())

    def test_local_sy_gate_one_qubit(self):
        a = self.random_uniform_mps(2, 1, truncate=True, normalize=True)
        U = LocalRotationsLayer(1, "Sy", default_parameters=[0.35])
        self.assertSimilar(
            U.apply(a).to_vector(), expm(-1j * 0.35 * self.Sy) @ a.to_vector()
        )

    def test_local_sy_gate_two_qubits_same_angle(self):
        a = self.random_uniform_mps(2, 2, truncate=True, normalize=True)
        U = LocalRotationsLayer(2, "Sy", same_parameter=True, default_parameters=[0.35])
        U1 = expm(-1j * 0.35 * self.Sy)
        U2 = expm(-1j * 0.35 * self.Sy)
        self.assertSimilar(U.apply(a).to_vector(), np.kron(U1, U2) @ a.to_vector())

    def test_local_sy_gate_two_qubits_different_angles(self):
        a = self.random_uniform_mps(2, 2, truncate=True, normalize=True)
        U = LocalRotationsLayer(2, "Sy", default_parameters=[0.35, 0.13])
        U1 = expm(-1j * 0.35 * self.Sy)
        U2 = expm(-1j * 0.13 * self.Sy)
        self.assertSimilar(U.apply(a).to_vector(), np.kron(U1, U2) @ a.to_vector())

    def test_local_sz_gate_one_qubit(self):
        a = self.random_uniform_mps(2, 1, truncate=True, normalize=True)
        U = LocalRotationsLayer(1, "Sz", default_parameters=[0.35])
        self.assertSimilar(
            U.apply(a).to_vector(), expm(-1j * 0.35 * self.Sz) @ a.to_vector()
        )

    def test_local_sz_gate_two_qubits_same_angle(self):
        a = self.random_uniform_mps(2, 2, truncate=True, normalize=True)
        U = LocalRotationsLayer(2, "Sz", same_parameter=True, default_parameters=[0.35])
        U1 = expm(-1j * 0.35 * self.Sz)
        U2 = expm(-1j * 0.35 * self.Sz)
        self.assertSimilar(U.apply(a).to_vector(), np.kron(U1, U2) @ a.to_vector())

    def test_local_sz_gate_two_qubits_different_angles(self):
        a = self.random_uniform_mps(2, 2, truncate=True, normalize=True)
        U = LocalRotationsLayer(2, "Sz", default_parameters=[0.35, 0.13])
        U1 = expm(-1j * 0.35 * self.Sz)
        U2 = expm(-1j * 0.13 * self.Sz)
        self.assertSimilar(U.apply(a).to_vector(), np.kron(U1, U2) @ a.to_vector())


class TestEntanglingLayerCircuit(TestCase):
    CNOT = interpret_operator("CNOT")
    CZ = interpret_operator("CZ")

    def test_entangling_layer_rejects_operators_not_acting_on_two_qubits(self):
        with self.assertRaises(Exception):
            TwoQubitGatesLayer(2, "Sx")

    def test_entangling_layer_apply_in_place_only_modifies_canonical_mps(self):
        a = self.random_uniform_mps(2, 2, truncate=True, normalize=True)
        U = TwoQubitGatesLayer(2, "CNOT")
        b = U.apply_inplace(a)
        self.assertTrue(a is b)
        a = MPS(a)
        b = U.apply_inplace(a)
        self.assertFalse(b is a)

    def test_local_gates_matmul_works(self):
        a = self.random_uniform_mps(2, 2, truncate=True, normalize=True)
        U = TwoQubitGatesLayer(2, "CNOT")
        b = a.copy()
        c = U.apply_inplace(a)
        d = U @ b
        self.assertSimilar(c, d)

    def test_entangling_layer_apply_works_with_other_centers(self):
        a = self.random_uniform_mps(2, 3, truncate=True, normalize=True)
        U = TwoQubitGatesLayer(3, "CNOT", direction=1)
        b = U.apply(CanonicalMPS(a, center=1))
        c = U.apply(CanonicalMPS(a, center=2))
        d = U.apply(CanonicalMPS(a, center=0))
        self.assertSimilar(b, c)
        self.assertSimilar(b, d)

        U = TwoQubitGatesLayer(3, "CNOT", direction=-1)
        b = U.apply(CanonicalMPS(a, center=1))
        c = U.apply(CanonicalMPS(a, center=2))
        d = U.apply(CanonicalMPS(a, center=0))
        self.assertSimilar(b, c)
        self.assertSimilar(b, d)

    def test_entangling_layer_apply_rejects_parameters(self):
        a = self.random_uniform_mps(2, 2, truncate=True, normalize=True)
        U = TwoQubitGatesLayer(2, "CNOT")
        with self.assertRaises(Exception):
            U.apply_inplace(a, [0, 0])

    def test_single_cnot_circuit(self):
        a = self.random_uniform_mps(2, 2, truncate=True, normalize=True)
        U = TwoQubitGatesLayer(2, "CNOT")
        self.assertSimilar(U.apply(a).to_vector(), self.CNOT @ a.to_vector())

    def test_two_cnot_circuit(self):
        a = self.random_uniform_mps(2, 3, truncate=True, normalize=True)
        U = TwoQubitGatesLayer(3, "CNOT")
        U1 = np.kron(self.CNOT, np.eye(2))
        U2 = np.kron(np.eye(2), self.CNOT)
        self.assertSimilar(U.apply(a).to_vector(), U2 @ U1 @ a.to_vector())

    def test_three_cnot_circuit(self):
        a = self.random_uniform_mps(2, 4, truncate=True, normalize=True)
        U = TwoQubitGatesLayer(4, "CNOT")
        U1 = np.kron(self.CNOT, np.eye(4))
        U2 = np.kron(np.kron(np.eye(2), self.CNOT), np.eye(2))
        U3 = np.kron(np.eye(4), self.CNOT)
        self.assertSimilar(U.apply(a).to_vector(), U3 @ U2 @ U1 @ a.to_vector())

    def test_single_cz_circuit(self):
        a = self.random_uniform_mps(2, 2, truncate=True, normalize=True)
        U = TwoQubitGatesLayer(2, "CZ")
        self.assertSimilar(U.apply(a).to_vector(), self.CZ @ a.to_vector())

    def test_two_cz_circuit(self):
        a = self.random_uniform_mps(2, 3, truncate=True, normalize=True)
        U = TwoQubitGatesLayer(3, "CZ")
        U1 = np.kron(self.CZ, np.eye(2))
        U2 = np.kron(np.eye(2), self.CZ)
        self.assertSimilar(U.apply(a).to_vector(), U2 @ U1 @ a.to_vector())

    def test_three_cz_circuit(self):
        a = self.random_uniform_mps(2, 4, truncate=True, normalize=True)
        U = TwoQubitGatesLayer(4, "CZ")
        U1 = np.kron(self.CZ, np.eye(4))
        U2 = np.kron(np.kron(np.eye(2), self.CZ), np.eye(2))
        U3 = np.kron(np.eye(4), self.CZ)
        self.assertSimilar(U.apply(a).to_vector(), U3 @ U2 @ U1 @ a.to_vector())


class TestVQE(TestCase):
    Sy = interpret_operator("Sy")
    CNOT = interpret_operator("CNOT")

    def test_VQE_two_qubits_one_layer(self):
        a = self.random_uniform_mps(2, 2, truncate=True, normalize=True)
        U = VQECircuit(2, 1, 2 * [0.13])
        U1 = np.kron(expm(-1j * 0.13 * self.Sy), expm(-1j * 0.13 * self.Sy))
        U2 = self.CNOT
        self.assertSimilar(U.apply(a).to_vector(), U2 @ U1 @ a.to_vector())

    def test_VQE_two_qubits_two_layers(self):
        a = self.random_uniform_mps(2, 2, truncate=True, normalize=True)
        U = VQECircuit(2, 2, [0.13, -0.5, -0.25, 0.73])
        U1 = np.kron(expm(-1j * 0.13 * self.Sy), expm(-1j * -0.5 * self.Sy))
        U2 = self.CNOT
        U3 = np.kron(expm(-1j * -0.25 * self.Sy), expm(-1j * 0.73 * self.Sy))
        U4 = self.CNOT
        self.assertSimilar(U.apply(a).to_vector(), U4 @ U3 @ U2 @ U1 @ a.to_vector())

    def test_VQE_entangling_layer_order(self):
        a = self.random_uniform_mps(2, 3, truncate=True, normalize=True)
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
        a = self.random_uniform_mps(2, 2, truncate=True, normalize=True)
        U = VQECircuit(2, 1, 2 * [0.0])
        U1 = np.kron(expm(-1j * 0.13 * self.Sy), expm(-1j * 0.15 * self.Sy))
        U2 = self.CNOT
        self.assertSimilar(
            U.apply(a, [0.13, 0.15]).to_vector(), U2 @ U1 @ a.to_vector()
        )
