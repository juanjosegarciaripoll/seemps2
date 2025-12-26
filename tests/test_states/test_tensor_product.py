import numpy as np
from seemps.state import (
    MPS,
    mps_tensor_product,
    product_state,
    DEFAULT_STRATEGY,
    NO_TRUNCATION,
)
from ..tools import TestCase


class TestMPSTensorProduct(TestCase):
    def setUp(self) -> None:
        v1 = np.array([1, -3j])
        v2 = np.array([-1, 0, -1])
        self.v1 = v1 / np.linalg.norm(v1)
        self.v2 = v2 / np.linalg.norm(v2)
        self.v1_tensor = self.v1.reshape(1, -1, 1)
        self.v2_tensor = self.v2.reshape(1, -1, 1)
        return super().setUp()

    def test_tensor_product_states_A_order(self):
        A = product_state(self.v1, 3)
        B = product_state(self.v2, 2)
        C = mps_tensor_product([A, B], "A", None)

        self.assertEqual(len(C), 3 + 2)
        self.assertEqual(C.physical_dimensions(), [2, 2, 2, 3, 3])
        self.assertSimilar(C.to_vector(), np.kron(A.to_vector(), B.to_vector()))
        self.assertEqual(C.bond_dimensions(), [1] * 6)

        self.assertSimilar(C[0], self.v1_tensor)
        self.assertSimilar(C[1], self.v1_tensor)
        self.assertSimilar(C[2], self.v1_tensor)
        self.assertSimilar(C[3], self.v2_tensor)
        self.assertSimilar(C[4], self.v2_tensor)

    def test_tensor_product_states_B_order(self):
        A = product_state(self.v1, 3)
        B = product_state(self.v2, 2)
        C = mps_tensor_product([A, B], "B", None)

        self.assertEqual(len(C), 3 + 2)
        self.assertEqual(C.physical_dimensions(), [2, 3, 2, 3, 2])
        self.assertEqual(C.bond_dimensions(), [1] * 6)

        self.assertSimilar(C[0], self.v1_tensor)
        self.assertSimilar(C[1], self.v2_tensor)
        self.assertSimilar(C[2], self.v1_tensor)
        self.assertSimilar(C[3], self.v2_tensor)
        self.assertSimilar(C[4], self.v1_tensor)

    def test_tensor_product_A_order(self):
        A_size = [2, 2]
        Av = self.rng.normal(size=A_size).reshape(-1)
        A = MPS.from_vector(Av, A_size, normalize=False)
        self.assertSimilar(A.to_vector(), Av)

        B_size = [3, 3, 3]
        Bv = self.rng.normal(size=B_size).reshape(-1)
        B = MPS.from_vector(Bv, B_size, normalize=False)
        self.assertSimilar(B.to_vector(), Bv)

        for strategy in [None, DEFAULT_STRATEGY, NO_TRUNCATION]:
            C = mps_tensor_product([A, B], "A", strategy)
            self.assertEqual(len(C), 3 + 2)
            self.assertEqual(C.physical_dimensions(), [2, 2, 3, 3, 3])
            self.assertEqual(A.bond_dimensions(), [1, 2, 1])
            self.assertEqual(B.bond_dimensions(), [1, 3, 3, 1])
            self.assertEqual(C.bond_dimensions(), [1, 2, 1, 3, 3, 1])
            if strategy is None:
                self.assertSimilar(C[0], A[0])
                self.assertSimilar(C[1], A[1])
                self.assertSimilar(C[2], B[0])
                self.assertSimilar(C[3], B[1])
                self.assertSimilar(C[4], B[2])

            AB = (Av.reshape(2, 2, 1, 1, 1) * Bv.reshape(1, 1, 3, 3, 3)).reshape(-1)
            self.assertSimilar(C.to_vector(), AB.reshape(-1))

    def test_tensor_product_B_order(self):
        A_size = [2, 2]
        Av = self.rng.normal(size=A_size).reshape(-1)
        A = MPS.from_vector(Av, A_size, normalize=False)
        self.assertSimilar(A.to_vector(), Av)

        B_size = [3, 3, 3]
        Bv = self.rng.normal(size=B_size).reshape(-1)
        B = MPS.from_vector(Bv, B_size, normalize=False)
        self.assertSimilar(B.to_vector(), Bv)

        for strategy in [None, DEFAULT_STRATEGY, NO_TRUNCATION]:
            C = mps_tensor_product([A, B], "B", strategy)
            self.assertEqual(len(C), 3 + 2)
            self.assertEqual(C.physical_dimensions(), [2, 3, 2, 3, 3])
            self.assertEqual(A.bond_dimensions(), [1, 2, 1])
            self.assertEqual(B.bond_dimensions(), [1, 3, 3, 1])
            self.assertEqual(C.bond_dimensions(), [1, 2, 6, 3, 3, 1])

            AB = (Av.reshape(2, 1, 2, 1, 1) * Bv.reshape(1, 3, 1, 3, 3)).reshape(-1)
            self.assertSimilar(C.to_vector(), AB.reshape(-1))
