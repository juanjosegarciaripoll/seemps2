import numpy as np
from seemps.state import NO_TRUNCATION, mps_tensor_sum
from ..test_analysis.tools_analysis import reorder_tensor
from ..tools import TestCase
from seemps.analysis.mesh import RegularInterval
from seemps.analysis.factories import mps_interval


class TestMPSOperations(TestCase):
    def test_tensor_sum(self):
        sites = 5
        interval = RegularInterval(-1, 2, 2**sites)
        mps_x = mps_interval(interval)
        X, Y = np.meshgrid(interval.to_vector(), interval.to_vector())
        # Order A
        mps_x_plus_y_A = mps_tensor_sum([mps_x, mps_x], mps_order="A")
        Z_mps_A = mps_x_plus_y_A.to_vector().reshape((2**sites, 2**sites))
        self.assertSimilar(Z_mps_A, X + Y)
        # Order B
        mps_x_plus_y_B = mps_tensor_sum([mps_x, mps_x], mps_order="B")
        Z_mps_B = mps_x_plus_y_B.to_vector().reshape((2**sites, 2**sites))
        Z_mps_B = reorder_tensor(Z_mps_B, [sites, sites])
        self.assertSimilar(Z_mps_B, X + Y)

    def test_tensor_sum_one_site(self):
        A = self.random_mps([2, 3, 4])
        self.assertSimilar(
            mps_tensor_sum([A], mps_order="A").to_vector(), A.to_vector()
        )

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
