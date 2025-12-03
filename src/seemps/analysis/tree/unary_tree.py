from __future__ import annotations

import numpy as np
from typing import Callable
from dataclasses import dataclass

from ...tools import make_logger, Logger
from ...typing import Vector, Matrix
from .branch import (
    BranchNode,
    propagate_images,
    get_transitions,
    assemble_cores,
)
from .sparse_mps import SparseMPS


class UnaryRootNode:
    """
    Root functional dependence for a `UnaryTree`.

    Represents a binary function f(x_L, x_s), where x_L arises from the left subtrees and x_s is
    selected from the given discretization grid, where the index s represents the physical dimensions
    of the MPS core.
    """

    # TODO: Type for vectorized arguments (looks cumbersome)
    def __init__(self, func: Callable, grid: Vector):
        self.func = func
        self.grid = grid
        self.N = len(grid)

    def evaluate(self, x_L: float | None, s: int) -> float:
        if x_L is None:
            return 0
        x_s = self.grid[s]
        return self.func(x_L, x_s)


@dataclass
class UnaryTree:
    """
    Unary-tree representation of a multivariate function.

    This class encodes a multivariate function with chain-like algebraic structure:

        f_n[...f_2(f_1(None, s_1), s_2), ..., s_n].

    The tree is composed of left nodes (`BranchNode`) representing the inner functional dependencies
    f_i and a right-most root node (`UnaryRootNode`) representing the binary function f_n that
    terminates the chain evaluation. Each node presents a free parameter x_s representing the input
    dimensions of the function (i.e., the leaves of the tree) and the physical dimensions of the
    MPS cores.

    This representation can be efficiently loaded into a MPS using :func:`mps_unary_tree`.

    Example
    -------
    A unary tree for the Heaviside function applied to the running sum

        f(x₁,x₂,... ) = Θ(x₁ + x₂ + ...)

    can be constructed as follows:

    - Each `BranchNode` represents the partial accumulation f_i(x_in, x_s) = x_in + x_s where
      its discretization grid is the domain of x_i and s ranges over its support.
    - The `UnaryRootNode` applies the Heaviside step f_n(x_{n-1}, x_s) = Θ(x_{n-1} + x_s).
    """

    left_nodes: list[BranchNode]
    root_node: UnaryRootNode

    def __post_init__(self):
        self.center = len(self.left_nodes)
        left_dimensions = [node.N for node in self.left_nodes]
        self.physical_dimensions = left_dimensions + [self.root_node.N]
        self.length = len(self.physical_dimensions)


def mps_unary_tree(
    unary_tree: UnaryTree, allowed_values: Vector | None = None
) -> SparseMPS:
    """
    Compute the MPS representation of a function encoded as a `UnaryTree`.

    Returns an `SparseMPS` where each core is highly sparse and represented by a collection of
    CSR matrices (see `SparseCore`). This construction is particularly efficient for functions with
    few distinct outputs, which are recompressed with a right-to-left binning sweep. Such outputs
    must be explicitly specified using the `allowed_values` parameter. For example, the Heaviside
    function Θ(f(x, y, z, ...)) has `allowed_values=[0, 1]`.
    Source: https://arxiv.org/abs/2206.03832

    Parameters
    ----------
    unary_tree : UnaryTree
        Unary tree representation of the target multivariate function.
    allowed_values : Vector, optional
        Discrete set of possible function outputs.

    Returns
    -------
    SparseMPS
        Sparse MPS approximation of the target multivariate function.
    """
    with make_logger(2) as logger:
        logger("Computing branch images:")
        left_images = propagate_images(unary_tree.left_nodes, logger)

        if allowed_values is not None:  # Recompression sweep (more efficient)
            logger("Computing grouped transitions:")
            root_transition, left_transitions = _recompress_transitions(
                unary_tree, left_images, allowed_values, logger
            )
        else:
            logger("Computing transitions:")
            left_transitions = get_transitions(
                unary_tree.left_nodes, left_images, logger
            )
            root_transition = {
                (k_L, s): unary_tree.root_node.evaluate(x_L, s)
                for k_L, x_L in enumerate(left_images[-1])
                for s in range(unary_tree.root_node.N)
            }

        logger("Computing MPS cores:")
        left_cores = assemble_cores(left_transitions, logger)
        # Compute root core
        coords = np.array(list(root_transition.keys()))
        values = np.array(list(root_transition.values()))
        χ_L = 1 + np.max(coords[:, 0])
        N = 1 + np.max(coords[:, 1])
        shape = (χ_L.item(), N.item(), 1)
        root_core = np.zeros(shape)
        root_core[tuple(coords.T)] = values.reshape(-1, 1)
        logger(f"Root node | Core of shape {shape}.")

    cores = left_cores + [root_core]
    return SparseMPS(cores)


def _recompress_transitions(
    unary_tree: UnaryTree,
    left_images: list[Vector],
    tensor_values: Vector,
    logger: Logger = Logger(),
) -> tuple[dict, list[dict]]:
    l = unary_tree.length - 1

    # Compute root node transition
    node = unary_tree.root_node
    x_in = left_images[-1]

    # Here we push vectorized arguments.
    # The type checker complains because the evaluate method is typed for scalar arguments.
    a = node.evaluate(x_in[:, np.newaxis], np.arange(node.N))  # type: ignore
    a = _round_matrix_to_vector(a, tensor_values)  # type: ignore
    root_transition = _build_transition(a)

    x_grouped = _group_x(x_in, a)
    logger(f"Node {l}/{l} | Image compressed ({len(x_in)} -> {len(x_grouped)}).")

    # Compute left node transitions
    left_transitions = []
    for i in reversed(range(l)):
        node = unary_tree.left_nodes[i]
        x_in = left_images[i]

        # Here we again push vectorized arguments
        a = node.evaluate(x_in[:, np.newaxis], np.arange(node.N))  # type: ignore
        A = _map_to_group(a, x_grouped)  # type: ignore
        transition = _build_transition(A)
        left_transitions.append(transition)

        x_grouped = _group_x(x_in, A)
        logger(f"Node {i}/{l} | Image compressed ({len(x_in)} -> {len(x_grouped)}).")

    return root_transition, left_transitions[::-1]


def _round_matrix_to_vector(matrix: Matrix, vector: Vector) -> Matrix:
    """Rounds elements in `matrix` to the nearest values in `vector`."""
    diff = np.abs(matrix[..., np.newaxis] - vector)
    indices = np.argmin(diff, axis=-1)
    return vector[indices]


def _group_x(x_in: Vector, a: Matrix) -> tuple[Vector]:
    """Groups elements of `x_in` based on unique rows in `a`."""
    # TODO: Optimize
    index_map = {}
    for index, row in enumerate(a):
        row_tuple = tuple(row)
        if row_tuple not in index_map:
            index_map[row_tuple] = np.array([x_in[index]])
        else:
            index_map[row_tuple] = np.append(index_map[row_tuple], x_in[index])
    return tuple(index_map.values())


def _build_transition(a: Matrix) -> dict:
    """Efficiently constructs a dictionary mapping (k, s) to unique values in matrix `a`."""
    unique_rows = _get_unique_rows(a)
    k_indices = np.arange(unique_rows.shape[0])
    s_indices = np.arange(unique_rows.shape[1])
    k_grid, s_grid = np.meshgrid(k_indices, s_indices, indexing="ij")

    keys = zip(k_grid.reshape(-1), s_grid.reshape(-1))
    values = unique_rows.reshape(-1)
    return {key: val for key, val in zip(keys, values)}


def _map_to_group(a: Matrix, x_grouped: tuple[Vector]) -> Matrix:
    """Maps values in `a` to group indices efficiently."""
    bounds = np.array([group[-1] for group in x_grouped])
    A = np.searchsorted(bounds, a, side="left")
    return np.minimum(A, len(x_grouped) - 1)


def _get_unique_rows(a: Matrix) -> Matrix:
    """Gets the unique rows of the matrix `a` without reordering."""
    _, idx = np.unique(a, axis=0, return_index=True)
    return a[np.sort(idx)]
