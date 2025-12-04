from __future__ import annotations
import numpy as np
from scipy.sparse import lil_array
from typing import Callable, overload
from ...tools import Logger, make_logger
from ...typing import Vector
from .sparse_mps import SparseCore


class BranchNode:
    def __init__(
        self,
        func: Callable,
        grid: Vector,
        binning_tol: float | None = None,
        max_rank: int | None = None,
    ):
        """
        Internal node in a computational tree.

        A `BranchNode` represents an intermediate functional dependence

            x_out = func(x_in, x_s),

        where:
            - x_in is a value passed by a child subtree (i.e., by an upstream functional dependence).
            - x_s is a free parameter obtained by evaluating the node's discretization grid at index s.

        For each call, the node receives an input value x_in from upstream and pairs it with a grid
        value x_s = grid[s] representing one possible configuration of the local variable.
        The contained function `func` is then evaluated on (x_in, x_s), where x_in may be None if
        the node is placed at the boundary and has no further upstream dependencies.
        The index s represents the local physical index of the MPS core generated at this node.

        The node presents compression parameters used to optionally compress the output image of
        `func` by merging together values within a relative tolerance (`binning_tol`) or by enforcing
        a maximum allowed MPS rank (`max_rank`).

        Parameters
        ----------
        func : Callable
            The functional dependence x_out = func(x_in, x_s).
        grid : Sequence
            Discretization of the local variable x_s. Its size defines the local physical dimension.
        binning_tol : float, optional
            Relative tolerance for binning similar output values.
        max_rank : int, optional
            Upper bound on the size of the compressed output image.
        """
        self.func = func
        self.grid = grid
        self.binning_tol = binning_tol
        self.max_rank = max_rank
        self.N = len(grid)

    @overload
    def evaluate(self, x_in: float | None, s: int) -> float | None: ...

    @overload
    def evaluate(self, x_in: np.ndarray | None, s: np.ndarray) -> np.ndarray | None: ...

    def evaluate(self, x_in, s):
        if x_in is None:
            return None
        x_s = self.grid[s]
        return self.func(x_in, x_s)

    def compute_image(
        self, values: Vector, default_tol: float = 1e-4, tol_multiplier: float = 1.25
    ) -> Vector:
        """
        Compute all distinct outputs produced by applying the node function to an input
        array `values` over all grid indices.

        The resulting image size is optionally compressed by binning similar values up to a maximal
        size `max_rank`. Returns a 1D array of sorted unique values.
        """
        logger = make_logger(3)

        # Compute the image. TODO: Vectorize
        image_matrix = np.zeros((len(values), self.N))
        for j, x_in in enumerate(values):
            for s in range(self.N):
                value = self.evaluate(x_in, s)
                image_matrix[j, s] = np.nan if value is None else value

        # Format the image
        image = image_matrix.reshape(-1)
        image = image[~np.isnan(image)]
        image = np.unique(image)
        logger(f"\tIncoming image of size {len(image)}.")

        # Compress the image
        if self.binning_tol is not None or self.max_rank is not None:
            binning_tol = default_tol if self.binning_tol is None else self.binning_tol
            image = self._bin_image(image, binning_tol)
            logger(
                f"\tImage compressed to {len(image)} bins with tolerance {binning_tol:.3e}."
            )

            if self.max_rank is not None:
                while len(image) > self.max_rank:
                    binning_tol *= tol_multiplier
                    image = self._bin_image(image, binning_tol)
                    logger(
                        f"\tImage compressed to {len(image)} bins with tolerance {binning_tol:.3e}."
                    )

        logger.close()
        return image

    @staticmethod
    def _bin_image(image: Vector, binning_tol: float) -> Vector:
        """Combines the values of the input `image` that are closer than `binning_tol` to reduce image size."""
        binned_image = []
        bin = [image[0]]
        for x in image[1:]:
            error = abs((x - bin[0]) / bin[0])
            if error <= binning_tol:
                bin.append(x)
            else:
                binned_image.append(np.mean(bin))
                bin = [x]
        binned_image.append(np.mean(bin))
        return np.array(binned_image)


def propagate_images(
    nodes: list[BranchNode], logger: Logger = Logger()
) -> list[Vector]:
    """Helper function to propagate an initial image through a sequence of BranchNodes."""
    l = len(nodes)
    images = [np.array([0.0])]
    for i, node in enumerate(nodes):
        image = node.compute_image(images[-1])
        logger(f"Node {(i + 1)}/{l} | Image of size {len(image)}.")
        images.append(image)
    return images


def get_transitions(
    nodes: list[BranchNode], images: list[Vector], logger: Logger = Logger()
) -> list[dict]:
    """
    Helper functions to construct the transition mappings for a chain of BranchNodes.

    These transitions are essential for assembling the MPS cores from the functional dependencies by
    determining their structur. Given consecutive images R_in and R_pout, the transition determines
    for each input value index k_in and grid index s, the corresponding output index k_out such that:

        x_out = func(R_in[k_in], grid[s])  ≈  R_out[k_out].

    Exact matches are used when possible; otherwise the nearest value R_out is chosen. The result for
    each node is a dictionary of transitions (k_in, s) -> k_out.
    """
    l = len(nodes)
    transitions = []
    for i, node in enumerate(nodes):
        R_in = images[i]
        R_out = images[i + 1]

        # Create lookup tables for fast O(1) search
        R_out_lookup = {value: idx for idx, value in enumerate(R_out)}

        transition = {}
        for s in range(node.N):
            for k_in, x_in in enumerate(R_in):
                x_out = node.evaluate(x_in, s)
                if x_out is not None:
                    k_out = R_out_lookup.get(x_out, None)
                    # If not found, find closest index in R_out with np.searchsorted
                    if k_out is None:
                        k_out = int(np.searchsorted(R_out, x_out, side="left"))
                        k_out = min(k_out, len(R_out) - 1)
                    transition[(k_in, s)] = k_out
        logger(f"Node {(i + 1)}/{l} | Transition of size {len(transition)}.")
        transitions.append(transition)

    return transitions


def assemble_cores(
    transitions: list[dict], logger: Logger = Logger()
) -> list[SparseCore]:
    """
    Helper function that assembles sparse MPS cores from the transition mappings.

    These MPS cores translate the transitions into tensor slices. For each node, a rank-3 tensor
    A[r_L, s, r_R] is assembled such that:

        A[k_in, s, k_out] = 1

    whenever the transition dictionary contains (k_in, s) -> k_out. Each physical slice (fixed s)
    is stored as a sparse CSR matrix, producing a compact MPS representation.
    """
    cores = []
    l = len(transitions)
    for i, transition in enumerate(transitions):
        coords = np.array([(k_in, s, k_out) for (k_in, s), k_out in transition.items()])
        χ_L = 1 + np.max(coords[:, 0])
        N = 1 + np.max(coords[:, 1])
        χ_R = 1 + np.max(coords[:, 2])

        data = [lil_array((χ_L, χ_R)) for _ in range(N)]
        for k_in, s, k_out in coords:
            data[s][k_in, k_out] += 1

        core = SparseCore([matrix.tocsr() for matrix in data])
        logger(f"Node {(i + 1)}/{l} | Core of shape {core.shape}.")
        cores.append(core)

    return cores
