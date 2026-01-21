import numpy as np
from seemps.state import MPS
from ..tools import TestCase
from seemps.register.circuit import LocalRotationsLayer, interpret_operator
from scipy.linalg import expm


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
        a = self.random_uniform_canonical_mps(2, 1, truncate=True, normalize=True)
        U = LocalRotationsLayer(1, "Sx", default_parameters=[0.35])
        b = U.apply_inplace(a)
        self.assertTrue(a is b)
        a = MPS(a)
        b = U.apply_inplace(a)
        self.assertFalse(b is a)

    def test_local_gates_matmul_works(self):
        a = self.random_uniform_canonical_mps(2, 1, truncate=True, normalize=True)
        U = LocalRotationsLayer(1, "Sx", default_parameters=[0.35])
        b = a.copy()
        c = U.apply_inplace(a)
        d = U @ b
        self.assertSimilar(c, d)

    def test_local_sx_gate_one_qubit(self):
        a = self.random_uniform_canonical_mps(2, 1, truncate=True, normalize=True)
        U = LocalRotationsLayer(1, "Sx", default_parameters=[0.35])
        self.assertSimilar(
            U.apply(a).to_vector(), expm(-1j * 0.35 * self.Sx) @ a.to_vector()
        )

    def test_local_sx_gate_two_qubits_same_angle(self):
        a = self.random_uniform_canonical_mps(2, 2, truncate=True, normalize=True)
        U = LocalRotationsLayer(2, "Sx", same_parameter=True, default_parameters=[0.35])
        U1 = expm(-1j * 0.35 * self.Sx)
        U2 = expm(-1j * 0.35 * self.Sx)
        self.assertSimilar(U.apply(a).to_vector(), np.kron(U1, U2) @ a.to_vector())

    def test_local_sx_gate_two_qubits_different_angles(self):
        a = self.random_uniform_canonical_mps(2, 2, truncate=True, normalize=True)
        U = LocalRotationsLayer(2, "Sx", default_parameters=[0.35, 0.13])
        U1 = expm(-1j * 0.35 * self.Sx)
        U2 = expm(-1j * 0.13 * self.Sx)
        self.assertSimilar(U.apply(a).to_vector(), np.kron(U1, U2) @ a.to_vector())

    def test_local_sy_gate_one_qubit(self):
        a = self.random_uniform_canonical_mps(2, 1, truncate=True, normalize=True)
        U = LocalRotationsLayer(1, "Sy", default_parameters=[0.35])
        self.assertSimilar(
            U.apply(a).to_vector(), expm(-1j * 0.35 * self.Sy) @ a.to_vector()
        )

    def test_local_sy_gate_two_qubits_same_angle(self):
        a = self.random_uniform_canonical_mps(2, 2, truncate=True, normalize=True)
        U = LocalRotationsLayer(2, "Sy", same_parameter=True, default_parameters=[0.35])
        U1 = expm(-1j * 0.35 * self.Sy)
        U2 = expm(-1j * 0.35 * self.Sy)
        self.assertSimilar(U.apply(a).to_vector(), np.kron(U1, U2) @ a.to_vector())

    def test_local_sy_gate_two_qubits_different_angles(self):
        a = self.random_uniform_canonical_mps(2, 2, truncate=True, normalize=True)
        U = LocalRotationsLayer(2, "Sy", default_parameters=[0.35, 0.13])
        U1 = expm(-1j * 0.35 * self.Sy)
        U2 = expm(-1j * 0.13 * self.Sy)
        self.assertSimilar(U.apply(a).to_vector(), np.kron(U1, U2) @ a.to_vector())

    def test_local_sz_gate_one_qubit(self):
        a = self.random_uniform_canonical_mps(2, 1, truncate=True, normalize=True)
        U = LocalRotationsLayer(1, "Sz", default_parameters=[0.35])
        self.assertSimilar(
            U.apply(a).to_vector(), expm(-1j * 0.35 * self.Sz) @ a.to_vector()
        )

    def test_local_sz_gate_two_qubits_same_angle(self):
        a = self.random_uniform_canonical_mps(2, 2, truncate=True, normalize=True)
        U = LocalRotationsLayer(2, "Sz", same_parameter=True, default_parameters=[0.35])
        U1 = expm(-1j * 0.35 * self.Sz)
        U2 = expm(-1j * 0.35 * self.Sz)
        self.assertSimilar(U.apply(a).to_vector(), np.kron(U1, U2) @ a.to_vector())

    def test_local_sz_gate_two_qubits_different_angles(self):
        a = self.random_uniform_canonical_mps(2, 2, truncate=True, normalize=True)
        U = LocalRotationsLayer(2, "Sz", default_parameters=[0.35, 0.13])
        U1 = expm(-1j * 0.35 * self.Sz)
        U2 = expm(-1j * 0.13 * self.Sz)
        self.assertSimilar(U.apply(a).to_vector(), np.kron(U1, U2) @ a.to_vector())
