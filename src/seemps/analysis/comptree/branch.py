from __future__ import annotations
import numpy as np
from scipy.sparse import lil_array
from typing import Callable, overload
from ...tools import Logger, make_logger
from ...typing import Vector
from .sparse_mps import SparseCore


class BranchNode:
    func: Callable
    grid: Vector
    binning_tol: float | None
    max_rank: int | None
    N: int

    def __init__(
        self,
        func: Callable,
        grid: Vector,
        binning_tol: float | None = None,
        max_rank: int | None = None,
    ):
        """
        Internal node in a computation-tree.

        A `BranchNode` represents an intermediate functional update

            x_out = func(x_in, x_s),

        where x_in is the value propagated from an upstream node and x_s is obtained by evaluating the
        node’s one-dimensional discretization grid at index s. The function `func` is applied to the
        pair (x_in, x_s) to produce the output passed downstream. If the node has no upstream dependency,
        x_in may be `None`.

        The index s labels the local physical dimension of the MPS core associated with this node.
        Optional compression parameters (`binning_tol`, `max_rank`) control the binning and truncation
        of the output image during MPS construction.

        Parameters
        ----------
        func : Callable
            Binary function implementing the update x_out = func(x_in, x_s).
        grid : Sequence
            One-dimensional discretization grid for the local variable. Its length denotes the local
            MPS physical dimension.
        binning_tol : float, optional
            Relative tolerance used to bin nearby output values during image compression.
        max_rank : int, optional
            Maximum allowed number of distinct values (bins) in the compressed output image,
            which bounds the maximum bond dimension of the MPS core generated at this node.
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

    def evaluate(self, x_in: float | np.ndarray | None, s: int | np.ndarray):
        if x_in is None:
            return None
        x_s = self.grid[s]
        return self.func(x_in, x_s)

    def compute_image(
        self, values: Vector, default_tol: float = 1e-4, tol_multiplier: float = 1.25
    ) -> Vector:
        """
        Compute the image of the node function over a set of input values.

        The node function x_out = func(x_in, x_s) is evaluated for all combinations of input
        values and grid indices, and the resulting outputs are collected into a set of sorted
        unique values. The image can be optionally compressed by binning nearby values and by
        enforcing `max_rank`, which prevents a combinatorial growth of image sizes across the
        tree and bounds the bond dimension of the corresponding MPS core.
        """
        logger = make_logger(3)

        # Compute the image
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
        """
        Bin nearby values in a sorted image using a relative tolerance.

        Consecutive image values within `binning_tol` are grouped and replaced by their mean,
        reducing the image size while controlling relative error.
        """
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
    Construct transition mappings for a chain of `BranchNode`s.

    For each node, this function determines how input image indices map to output image
    indices under the node's functional update. Given consecutive images R_in and R_out,
    the transition assigns, for each input index k_in and grid index s, the output index
    k_out such that

        func(R_in[k_in], grid[s]) ≈ R_out[k_out].

    When no binning is applied, exact matches in R_out are guaranteed. If binning is used,
    the nearest value in R_out is assigned instead. Each node yields a dictionary mapping
    (k_in, s) → k_out, which is used to assemble the corresponding MPS core.
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
    Assemble sparse MPS cores from transition mappings.

    For each node, a rank-3 tensor A[r_L, s, r_R] is constructed such that

        A[k_in, s, k_out] = 1

    whenever the transition mapping contains (k_in, s) → k_out. Each physical slice
    (fixed s) is stored as a sparse CSR matrix, yielding a compact sparse-core representation.
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
