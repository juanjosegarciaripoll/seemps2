import numpy as np
from ..tools import TestCase

from seemps.analysis.mesh import RegularInterval
from seemps.analysis.comptree import (
    mps_chain_tree,
    mps_binary_tree,
    ChainTree,
    ChainRoot,
    BinaryTree,
    BinaryRoot,
    BranchNode,
)
from seemps.analysis.comptree.branch import get_transitions


class TestBranchNode(TestCase):
    def test_binning_reduces_image(self):
        x = np.linspace(0, 1, 200)
        node = BranchNode(lambda _, x_s: np.sin(100 * x_s), grid=x, binning_tol=1e-2)
        image = node.compute_image(np.array([0.0]))
        self.assertLess(len(image), len(x))

    def test_max_rank_enforced(self):
        x = np.linspace(0, 1, 200)
        node = BranchNode(lambda _, x_s: x_s, grid=x, max_rank=10)
        image = node.compute_image(np.array([0.0]))
        self.assertLessEqual(len(image), 10)

    def test_exact_transitions_without_binning(self):
        x = np.linspace(0, 1, 200)
        node = BranchNode(lambda x_in, x_s: x_in + x_s, grid=x)
        images = [np.array([0.0]), x]
        transitions = get_transitions([node], images)
        for (_, s), k_out in transitions[0].items():
            self.assertEqual(images[1][k_out], x[s])


class TestChainTree(TestCase):
    def test_chain_tree_single_node(self):
        N = 50
        x = RegularInterval(-1.0, 1.0, N).to_vector()
        root_node = ChainRoot(lambda _, x_s: x_s**2, grid=x)
        tree = ChainTree([], root_node)
        mps = mps_chain_tree(tree)
        f_mps = mps.to_vector()
        self.assertTrue(np.allclose(f_mps, x**2))

    def test_2d_round_indicator(self):
        """
        Verify that a ChainTree can exactly encode the two-dimensional Heaviside indicator
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
        root_node = ChainRoot(
            lambda x_in, c: np.heaviside(c - x_in, 1.0),
            grid=np.array([1.0]),
        )

        tree = ChainTree(left_nodes, root_node)
        mps = mps_chain_tree(tree, allowed_support=np.array([0.0, 1.0]))
        Θ_mps = mps.to_vector().reshape(N, N)

        X1, X2 = np.meshgrid(x, x, indexing="ij")
        radius2 = X1**2 + X2**2
        Θ_dense = (radius2 <= 1.0).astype(float)

        self.assertTrue(np.allclose(Θ_mps, Θ_dense, atol=1e-12))


class TestBinaryTree(TestCase):
    def test_binary_tree_asymmetric(self):
        N = 50
        x = RegularInterval(-1.0, 1.0, N).to_vector()
        left_nodes = [BranchNode(lambda _, x_s: x_s, grid=x)]  # X1
        right_nodes = [
            BranchNode(lambda _, x_s: x_s**2, grid=x),  # X3
            BranchNode(lambda x_in, x_s: x_in * x_s, grid=x),  # X2
        ]
        root_node = BinaryRoot(lambda x_L, _, x_R: x_L - x_R, grid=np.array([0.0]))
        tree = BinaryTree(left_nodes, root_node, right_nodes)
        mps = mps_binary_tree(tree)
        F_mps = mps.to_vector().reshape(N, N, N)
        X1, X2, X3 = np.meshgrid(x, x, x, indexing="ij")
        F_dense = X1 - X2 * X3**2
        self.assertTrue(np.allclose(F_mps, F_dense))

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
        root_node = BinaryRoot(
            lambda x_L, _, x_R: x_L + x_R + x_L * x_R,
            grid=np.array([0.0]),
        )
        tree = BinaryTree(left_nodes, root_node, right_nodes)
        mps = mps_binary_tree(tree)
        F_mps = mps.to_vector().reshape(N, N)

        X1, X2 = np.meshgrid(x, x, indexing="ij")
        F_dense = X1 + X2 + X1 * X2
        self.assertTrue(np.allclose(F_mps, F_dense, atol=1e-12))
