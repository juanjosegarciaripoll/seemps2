from typing import Any
import numpy as np
from seemps.state import MPS, NO_TRUNCATION, mps_tensor_sum
from ..tools import SeeMPSTestCase


class TestMPSOperations(SeeMPSTestCase):
    def test_tensor_sum_one_site(self):
        A = self.random_mps([2, 3, 4])
        B = mps_tensor_sum([A], mps_order="A", strategy=None)
        self.assertEqual(A.physical_dimensions(), B.physical_dimensions())
        self.assertTrue(all(np.all(A[i] == B[i]) for i in range(len(A))))

        B = mps_tensor_sum([A], mps_order="B", strategy=None)
        self.assertEqual(A.physical_dimensions(), B.physical_dimensions())
        self.assertTrue(all(np.all(A[i] == B[i]) for i in range(len(A))))

    def test_tensor_sum_small_size_A_order(self):
        A = self.random_mps([2, 3, 4])
        B = self.random_mps([5, 3, 2])
        AB = mps_tensor_sum([A, B], mps_order="A", strategy=NO_TRUNCATION)
        A_v = A.to_vector()
        B_v = B.to_vector()
        self.assertSimilar(
            AB.to_vector(),
            np.kron(A.to_vector(), np.ones(B_v.shape))
            + np.kron(np.ones(A_v.shape), B.to_vector()),
        )

    def test_tensor_sum_small_size_B_order(self):
        A = self.random_mps([2, 3, 4])
        B = self.random_mps([2, 3, 4])
        AB = mps_tensor_sum([A, B], mps_order="B", strategy=NO_TRUNCATION)
        AB_t = AB.to_vector().reshape([2, 2, 3, 3, 4, 4])
        AB_v = AB_t.transpose([0, 2, 4, 1, 3, 5]).reshape(-1)
        A_v = A.to_vector()
        B_v = B.to_vector()
        self.assertSimilar(
            AB_v,
            np.kron(A.to_vector(), np.ones(B_v.shape))
            + np.kron(np.ones(A_v.shape), B.to_vector()),
        )

    def test_tensor_sum_small_size_B_order_different_sizes(self):
        A = self.random_mps([2, 3, 4])
        B = self.random_mps([6, 3, 4])
        AB = mps_tensor_sum([A, B], mps_order="B", strategy=NO_TRUNCATION)
        AB_t = AB.to_vector().reshape([2, 6, 3, 3, 4, 4])
        AB_v = AB_t.transpose([0, 2, 4, 1, 3, 5]).reshape(-1)
        A_v = A.to_vector()
        B_v = B.to_vector()
        self.assertSimilar(
            AB_v,
            np.kron(A.to_vector(), np.ones(B_v.shape))
            + np.kron(np.ones(A_v.shape), B.to_vector()),
        )

    def test_tensor_sum_single_site_terms_A_order(self):
        A = MPS.from_vector(np.array([1.0, 2.0j]), [2], normalize=False)
        B = MPS.from_vector(np.array([-1.0, 0.5]), [2], normalize=False)
        C = mps_tensor_sum([A, B], mps_order="A", strategy=NO_TRUNCATION)
        expected = np.kron(A.to_vector(), np.ones(2)) + np.kron(np.ones(2), B.to_vector())
        self.assertSimilar(C.to_vector(), expected)

    def test_tensor_sum_three_states_B_order_with_step_simplification(self):
        A = MPS.from_vector(np.array([1.0, 2.0j]), [2], normalize=False)
        B = MPS.from_vector(np.array([1.0, -1.0j, 2.0]), [3], normalize=False)
        C = MPS.from_vector(np.array([0.5, -2.0]), [2], normalize=False)
        result = mps_tensor_sum(
            [A, B, C], mps_order="B", strategy=NO_TRUNCATION, simplify_steps=True
        )
        expected = (
            np.kron(np.kron(A.to_vector(), np.ones(3)), np.ones(2))
            + np.kron(np.kron(np.ones(2), B.to_vector()), np.ones(2))
            + np.kron(np.kron(np.ones(2), np.ones(3)), C.to_vector())
        )
        self.assertEqual(result.physical_dimensions(), [2, 3, 2])
        self.assertSimilar(result.to_vector(), expected)

    def test_tensor_sum_rejects_invalid_order(self):
        A = self.random_mps([2, 3])
        invalid_order: Any = "invalid"
        with self.assertRaises(ValueError):
            mps_tensor_sum([A, A], mps_order=invalid_order, strategy=NO_TRUNCATION)
