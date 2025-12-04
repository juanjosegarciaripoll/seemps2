from __future__ import annotations
import numpy as np
from typing import Callable
import dataclasses
from ...tools import make_logger
from ...typing import Vector
from .branch import (
    BranchNode,
    propagate_images,
    get_transitions,
    assemble_cores,
)
from .sparse_mps import SparseMPS


class BinaryRootNode:
    """
    Root functional dependence for a `BinaryTree`.

    Represents a ternary function f(x_L, x_s, x_R), where x_L and x_R arise from the left and right
    subtrees and x_s is selected from the given discretization grid, where the index s represents
    the physical dimensions of the MPS core.
    """

    def __init__(self, func: Callable, grid: Vector):
        self.func = func
        self.grid = grid
        self.N = len(grid)

    def evaluate(self, x_L: float | None, s: int, x_R: float | None) -> float:
        if x_L is None or x_R is None:
            return 0
        x_s = self.grid[s]
        return self.func(x_L, x_s, x_R)


@dataclasses.dataclass
class BinaryTree:
    """
    Binary-tree representation of a multivariate function.

    This class encodes a multivariate function with a branching algebraic structure:

        f[ g1( g11(...), g12(...)) , g2( g21(...), g22(...) )].

    The tree is composed of left and right chains of `BranchNode`s, which define how input
    variables are combined within each subtree, and a `BinaryRootNode` that merges the two
    aggregated results through a ternary function. Each node exposes a grid variable x_s,
    corresponding to one input dimension and, ultimately, one physical dimension of the MPS.

    This can express hierarchical dependencies in a way that can be efficiently compiled into
    an MPS using :func:`mps_binary_tree`.
    """

    left_nodes: list[BranchNode]
    root_node: BinaryRootNode
    right_nodes: list[BranchNode]

    # Keep type checker happy
    center: int = dataclasses.field(init=False)
    physical_dimensions: list[int] = dataclasses.field(init=False)
    length: int = dataclasses.field(init=False)

    def __post_init__(self):
        self.center = len(self.left_nodes)
        left_dimensions = [node.N for node in self.left_nodes]
        right_dimensions = [node.N for node in self.right_nodes]
        self.physical_dimensions = (
            left_dimensions + [self.root_node.N] + right_dimensions
        )
        self.length = len(self.physical_dimensions)


def mps_binary_tree(binary_tree: BinaryTree) -> SparseMPS:
    """
    Compute the MPS representation of a function encoded as a `BinaryTree`.

    Returns an `SparseMPS` where each core is highly sparse and represented by a collection of
    CSR matrices (see `SparseCore`). It efficiently approximates the multivariate function spanned
    by the binary tree. Source: https://arxiv.org/abs/2206.03832

    Parameters
    ----------
    binary_tree : BinaryTree
        Binary tree representation of the target multivariate function.

    Returns
    -------
    SparseMPS
        Sparse MPS approximation of the target multivariate function.
    """

    with make_logger(2) as logger:
        logger("Computing branch images:")
        left_images = propagate_images(binary_tree.left_nodes, logger)
        right_images = propagate_images(binary_tree.right_nodes, logger)

        logger("Computing transitions:")
        left_transitions = get_transitions(binary_tree.left_nodes, left_images, logger)
        right_transitions = get_transitions(
            binary_tree.right_nodes, right_images, logger
        )
        root_transition = {
            (k_L, s, k_R): binary_tree.root_node.evaluate(x_L, s, x_R)
            for k_L, x_L in enumerate(left_images[-1])
            for s in range(binary_tree.root_node.N)
            for k_R, x_R in enumerate(right_images[-1])
        }

        logger("Computing MPS cores:")
        left_cores = assemble_cores(left_transitions, logger)
        right_cores = assemble_cores(right_transitions, logger)
        right_cores = [A.transpose() for A in right_cores][::-1]
        # Compute root core
        coords = np.array(list(root_transition.keys()))
        values = np.array(list(root_transition.values()))
        χ_L = 1 + np.max(coords[:, 0])
        N = 1 + np.max(coords[:, 1])
        χ_R = 1 + np.max(coords[:, 2])
        shape = (χ_L, N, χ_R)
        root_core = np.zeros(shape)
        root_core[tuple(coords.T)] = values
        logger(f"Center core of shape {shape}.")

    cores = left_cores + [root_core] + right_cores
    return SparseMPS(cores)
