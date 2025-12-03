import numpy as np
from ..tools import TestCase

from seemps.analysis.mesh import RegularInterval
from seemps.analysis.tree import (
    mps_unary_tree,
    mps_binary_tree,
    UnaryTree,
    UnaryRootNode,
    BinaryTree,
    BinaryRootNode,
    BranchNode,
)


class TestUnaryTree(TestCase):
    def test_2d_round_indicator(self):
        """
        Verify that a UnaryTree can exactly encode the two-dimensional Heaviside indicator
            Θ(x1, x2) = 1  if  x1^2 + x2^2 ≤ 1
                        0  otherwise.
        """
        N = 100
        interval = RegularInterval(-1.0, 1.0, N)
        x = interval.to_vector()

        weights = np.array([1.0, 1.0])
        left_nodes = [
            BranchNode(lambda x_in, x_s: x_in + weights[0] * x_s**2, grid=x),
            BranchNode(lambda x_in, x_s: x_in + weights[1] * x_s**2, grid=x),
        ]
        root_node = UnaryRootNode(
            lambda x_in, c: np.heaviside(c - x_in, 1.0),
            grid=np.array([1.0]),
        )

        tree = UnaryTree(left_nodes, root_node)
        mps = mps_unary_tree(tree, allowed_values=np.array([0.0, 1.0]))
        Θ_mps = mps.to_vector().reshape(N, N)

        X1, X2 = np.meshgrid(x, x, indexing="ij")
        radius2 = X1**2 + X2**2
        Θ_dense = (radius2 <= 1.0).astype(float)

        self.assertTrue(np.allclose(Θ_mps, Θ_dense, atol=1e-12))


class TestBinaryTree(TestCase):
    def test_sum_function(self):
        """
        Verify that a binary tree can encode the very basic function
            F(x1, x2) = x1 + x2 + x1 x2
        using two BranchNodes and one (empty) central binary root node.
        """
        N = 100
        interval = RegularInterval(-1.0, 1.0, N)
        x = interval.to_vector()

        left_nodes = [BranchNode(lambda _, x_s: x_s, grid=x)]
        right_nodes = [BranchNode(lambda _, x_s: x_s, grid=x)]

        root_node = BinaryRootNode(
            lambda x_L, _, x_R: x_L + x_R + x_L * x_R,
            grid=np.array([0.0]),
        )
        tree = BinaryTree(left_nodes, root_node, right_nodes)
        mps = mps_binary_tree(tree)
        F_mps = mps.to_vector().reshape(N, N)

        X1, X2 = np.meshgrid(x, x, indexing="ij")
        F_dense = X1 + X2 + X1 * X2

        self.assertTrue(np.allclose(F_mps, F_dense, atol=1e-12))
