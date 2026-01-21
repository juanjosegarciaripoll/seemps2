import numpy as np
from seemps.state import MPS, CanonicalMPS
from ..tools import TestCase
from seemps.register.circuit import TwoQubitGatesLayer, interpret_operator


class TestEntanglingLayerCircuit(TestCase):
    CNOT = interpret_operator("CNOT")
    CZ = interpret_operator("CZ")

    def test_entangling_layer_rejects_operators_not_acting_on_two_qubits(self):
        with self.assertRaises(Exception):
            TwoQubitGatesLayer(2, "Sx")

    def test_entangling_layer_apply_in_place_only_modifies_canonical_mps(self):
        a = self.random_uniform_canonical_mps(2, 2, truncate=True, normalize=True)
        U = TwoQubitGatesLayer(2, "CNOT")
        b = U.apply_inplace(a)
        self.assertTrue(a is b)
        a = MPS(a)
        b = U.apply_inplace(a)
        self.assertFalse(b is a)

    def test_local_gates_matmul_works(self):
        a = self.random_uniform_canonical_mps(2, 2, truncate=True, normalize=True)
        U = TwoQubitGatesLayer(2, "CNOT")
        b = a.copy()
        c = U.apply_inplace(a)
        d = U @ b
        self.assertSimilar(c, d)

    def test_entangling_layer_apply_works_with_other_centers(self):
        a = self.random_uniform_canonical_mps(2, 3, truncate=True, normalize=True)
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
        a = self.random_uniform_canonical_mps(2, 2, truncate=True, normalize=True)
        U = TwoQubitGatesLayer(2, "CNOT")
        with self.assertRaises(Exception):
            U.apply_inplace(a, [0, 0])

    def test_single_cnot_circuit(self):
        a = self.random_uniform_canonical_mps(2, 2, truncate=True, normalize=True)
        U = TwoQubitGatesLayer(2, "CNOT")
        self.assertSimilar(U.apply(a).to_vector(), self.CNOT @ a.to_vector())

    def test_two_cnot_circuit(self):
        a = self.random_uniform_canonical_mps(2, 3, truncate=True, normalize=True)
        U = TwoQubitGatesLayer(3, "CNOT")
        U1 = np.kron(self.CNOT, np.eye(2))
        U2 = np.kron(np.eye(2), self.CNOT)
        self.assertSimilar(U.apply(a).to_vector(), U2 @ U1 @ a.to_vector())

    def test_three_cnot_circuit(self):
        a = self.random_uniform_canonical_mps(2, 4, truncate=True, normalize=True)
        U = TwoQubitGatesLayer(4, "CNOT")
        U1 = np.kron(self.CNOT, np.eye(4))
        U2 = np.kron(np.kron(np.eye(2), self.CNOT), np.eye(2))
        U3 = np.kron(np.eye(4), self.CNOT)
        self.assertSimilar(U.apply(a).to_vector(), U3 @ U2 @ U1 @ a.to_vector())

    def test_single_cz_circuit(self):
        a = self.random_uniform_canonical_mps(2, 2, truncate=True, normalize=True)
        U = TwoQubitGatesLayer(2, "CZ")
        self.assertSimilar(U.apply(a).to_vector(), self.CZ @ a.to_vector())

    def test_two_cz_circuit(self):
        a = self.random_uniform_canonical_mps(2, 3, truncate=True, normalize=True)
        U = TwoQubitGatesLayer(3, "CZ")
        U1 = np.kron(self.CZ, np.eye(2))
        U2 = np.kron(np.eye(2), self.CZ)
        self.assertSimilar(U.apply(a).to_vector(), U2 @ U1 @ a.to_vector())

    def test_three_cz_circuit(self):
        a = self.random_uniform_canonical_mps(2, 4, truncate=True, normalize=True)
        U = TwoQubitGatesLayer(4, "CZ")
        U1 = np.kron(self.CZ, np.eye(4))
        U2 = np.kron(np.kron(np.eye(2), self.CZ), np.eye(2))
        U3 = np.kron(np.eye(4), self.CZ)
        self.assertSimilar(U.apply(a).to_vector(), U3 @ U2 @ U1 @ a.to_vector())
