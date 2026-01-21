from seemps.state import NO_TRUNCATION
from seemps.operators import MPO
from seemps.register.circuit import ParameterFreeMPO, interpret_operator
from ..tools import TestCase


class TestMPOLayerCircuit(TestCase):
    def setUp(self) -> None:
        super().setUp()
        sx = interpret_operator("SX")
        self.U = MPO.from_local_operators([sx] * 4, strategy=NO_TRUNCATION)

    def test_mpo_gates_accepts_mpo(self):
        ParameterFreeMPO(self.U)

    def test_mpo_gates_accepts_mpo_list(self):
        ParameterFreeMPO(self.U @ self.U)

    def test_mpo_gates_requires_qubit_operators(self):
        ParameterFreeMPO(self.U)
        with self.assertRaises(Exception):
            spin_1_op = interpret_operator("SX(1)")
            ParameterFreeMPO(MPO.from_local_operators([spin_1_op] * 4))

    def test_mpo_gates_equal_to_mpo_application(self):
        a = self.random_uniform_canonical_mps(2, self.U.size, truncate=True, normalize=True)
        U = ParameterFreeMPO(self.U)
        b = a.copy()
        c = U.apply_inplace(a)
        d = self.U @ b
        self.assertSimilar(c, d)

    def test_mpo_gates_equal_to_mpo_list_application(self):
        a = self.random_uniform_canonical_mps(2, self.U.size, truncate=True, normalize=True)
        U2 = ParameterFreeMPO(self.U @ self.U)
        b = a.copy()
        c = U2.apply_inplace(a)
        d = self.U @ (self.U @ b)
        self.assertSimilar(c, d)
