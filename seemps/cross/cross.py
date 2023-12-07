from copy import copy, deepcopy
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np

from ..state import MPS, random_mps
from ..tools import log
from ..truncate import simplify
from .maxvol import maxvol_rct, maxvol_sqr
from .mesh import Mesh


@dataclass
class CrossStrategy:
    """Parameters for the Tensor Cross Interpolation algorithm.

    Parameters
    ----------
    tol : float
        Maximum error allowed by the algorithm.
    maxiter : int
        Maximum number of sweeps allowed by the algorithm.
    maxrank : int
        Maximum bond dimension of the MPS allowed by the algorithm.
    mps_ordering : str
        Site ordering of the underlying MPS (assumed binary). Either "A" or "B".
    maxvol_sqr_tau : float
        Sensibility parameter for the square maxvol algorithm.
    maxvol_sqr_maxiter : int
        Maximum iterations for the square maxvol algorithm.
    maxvol_rct_tau : float
        Sensibility parameter for the rectangular maxvol algorithm.
    maxvol_rct_minrank : int
        Minimum allowed rank change introduced by the rectangular maxvol algorithm.
    maxvol_rct_maxrank : int
        Maximum allowed rank change introduced by the rectangular maxvol algorithm.
    error_type : str
        Method used to measure the error of the algorithm. Either "sampling" or "norm".
    """

    tol: float = 1e-10
    maxiter: int = 100
    maxrank: int = 100
    mps_ordering: str = "A"
    maxvol_sqr_tau: float = 1.05
    maxvol_sqr_maxiter: int = 100
    maxvol_rct_tau: float = 1.10
    maxvol_rct_minrank: int = 1
    maxvol_rct_maxrank: int = 1
    error_type: str = "sampling"


@dataclass
class Cross:
    """Cross class.

    This implements a data structure for the Tensor Cross Interpolation algorithm,
    which encodes a function defined on a discretized mesh on a Matrix Product
    State (MPS) by means of the skeleton decomposition.

    Parameters
    ----------
    func : Callable
        A multidimensional **vector-valued** function to be encoded in a MPS.
    mesh : Mesh
        A multidimensional discretized mesh on which the function is defined, defining
        an implicit tensor.
    mps : MPS
        An initial MPS with the same size as the mesh to serve as an initial approximation
        for the algorithm. Can be of 'binary' ([2] * dims * n) or 'tt' ([2**n] * dims) structure.
    cross_strategy : CrossStrategy
        An object which contains the algorithm parameters.
    """

    func: Callable
    mesh: Mesh
    mps: MPS
    cross_strategy: CrossStrategy

    def __post_init__(self):
        shape_mps = tuple(self.mps.physical_dimensions())
        shape_mesh = self.mesh.shape()[:-1]
        if np.prod(shape_mps) == np.prod(shape_mesh) and all(
            dim == 2 for dim in shape_mps
        ):
            self.structure = "binary"
        elif shape_mps == shape_mesh:
            self.structure = "tt"
        else:
            raise ValueError("Non-matching mesh and initial MPS")
        self.sites = len(self.mps)
        self.qubits = [int(np.log2(s)) for s in shape_mesh]


def _initialize(cross: Cross) -> None:
    """Initializes the Cross dataclass multi-indices and attributes executing a presweep
    on the initial MPS without evaluating the function and using the square maxvol algorithm
    without rank increments."""
    cross.I_physical = [
        np.arange(k, dtype=int).reshape(-1, 1) for k in cross.mps.physical_dimensions()
    ]
    cross.I_forward = [None for _ in range(cross.sites + 1)]
    cross.I_backward = [None for _ in range(cross.sites + 1)]
    cross.error = 1
    cross.sweep = 0
    cross.maxrank = 0

    cross_initial = copy(cross)
    cross_initial.cross_strategy = CrossStrategy(
        maxvol_rct_minrank=0, maxvol_rct_maxrank=0
    )

    # Forward pass
    R = np.ones((1, 1))
    for j in range(cross.sites):
        fiber = np.tensordot(R, cross.mps[j], 1)
        cross.mps[j], cross.I_forward[j + 1], R = _skeleton(
            fiber, cross_initial, j, ltr=True
        )
    cross.mps[cross.sites - 1] = np.tensordot(cross.mps[cross.sites - 1], R, 1)

    # Backward pass
    R = np.ones((1, 1))
    for j in range(cross.sites - 1, -1, -1):
        fiber = np.tensordot(cross.mps[j], R, 1)
        cross.mps[j], cross.I_backward[j], R = _skeleton(
            fiber, cross_initial, j, ltr=False
        )
    cross.mps[0] = np.tensordot(R, cross.mps[0], 1)


def _sweep(cross: Cross) -> None:
    """Runs a forward-backward sweep on the MPS that iteratively updates its tensors and
    forward / backward multi-indices (pivots) by means of the skeleton decomposition.
    """
    # Forward pass
    R = np.ones((1, 1))
    for j in range(cross.sites):
        fiber = _sample(cross, j)
        cross.mps[j], cross.I_forward[j + 1], R = _skeleton(fiber, cross, j, ltr=True)
    cross.mps[cross.sites - 1] = np.tensordot(cross.mps[cross.sites - 1], R, 1)

    # Backward pass
    R = np.ones((1, 1))
    for j in range(cross.sites - 1, -1, -1):
        fiber = _sample(cross, j)
        cross.mps[j], cross.I_backward[j], R = _skeleton(fiber, cross, j, ltr=False)
    cross.mps[0] = np.tensordot(R, cross.mps[0], 1)

    cross.sweep += 1
    cross.maxrank = max(cross.mps.bond_dimensions())


# TODO: Clean and optimize
def _skeleton(
    fiber: np.ndarray,
    cross: Cross,
    j: int,
    ltr: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns the skeleton decomposition of a tensor fiber using either the square
    or rectangular (rank-increasing) maxvol algorithm.
    """
    r1, s, r2 = fiber.shape

    # Reshape the fiber into a matrix
    if ltr:
        i_virtual = cross.I_forward[j]
        fiber = np.reshape(fiber, (r1 * s, r2), order="F")
    else:
        i_virtual = cross.I_backward[j + 1]
        fiber = np.reshape(fiber, (r1, s * r2), order="F").T

    # Perform QR factorization
    Q, R = np.linalg.qr(fiber)

    # Perform maxvol decomposition on the Q-factor
    if Q.shape[0] <= Q.shape[1]:
        i_maxvol = np.arange(Q.shape[0], dtype=int)
        Q_maxvol = np.eye(Q.shape[0], dtype=float)
    elif cross.cross_strategy.maxvol_rct_maxrank == 0:
        i_maxvol, Q_maxvol = maxvol_sqr(
            Q,
            k=cross.cross_strategy.maxvol_sqr_maxiter,
            e=cross.cross_strategy.maxvol_sqr_tau,
        )
    else:
        i_maxvol, Q_maxvol = maxvol_rct(
            Q,
            k=cross.cross_strategy.maxvol_sqr_maxiter,
            e=cross.cross_strategy.maxvol_sqr_tau,
            tau=cross.cross_strategy.maxvol_rct_tau,
            min_r=cross.cross_strategy.maxvol_rct_minrank,
            max_r=cross.cross_strategy.maxvol_rct_maxrank,
        )

    # Redefine the fiber as the decomposed Q-factor
    i_physical = cross.I_physical[j]
    if ltr:
        i_physical_ext = np.kron(i_physical, np.ones((r1, 1), dtype=int))
        fiber = np.reshape(Q_maxvol, (r1, s, -1), order="F")
        R = Q[i_maxvol, :] @ R
    else:
        i_physical_ext = np.kron(np.ones((r2, 1), dtype=int), i_physical)
        fiber = np.reshape(Q_maxvol.T, (-1, s, r2), order="F")
        R = (Q[i_maxvol, :] @ R).T

    # Redefine the maxvol indices in terms of the tensor multi-indices
    if i_virtual is not None:
        i_virtual_ext = (
            np.kron(np.ones((s, 1), dtype=int), i_virtual)
            if ltr
            else np.kron(i_virtual, np.ones((s, 1), dtype=int))
        )
        i_physical_ext = (
            np.hstack((i_virtual_ext, i_physical_ext))
            if ltr
            else np.hstack((i_physical_ext, i_virtual_ext))
        )
    i_maxvol = i_physical_ext[i_maxvol, :]

    return fiber, i_maxvol, R


def _sample(cross: Cross, j: int) -> np.ndarray:
    """Returns a fiber of the implicit tensor along the site j by evaluating the function at the
    kronecker product of the physical indices and the forward and backward maxvol multi-indices
    (pivots) that are present at that site.
    """
    i_physical = cross.I_physical[j]
    i_forward = cross.I_forward[j]
    i_backward = cross.I_backward[j + 1]
    r1 = i_forward.shape[0] if i_forward is not None else 1
    s = i_physical.shape[0]
    r2 = i_backward.shape[0] if i_backward is not None else 1
    indices = np.kron(
        np.kron(np.ones((r2, 1), dtype=int), i_physical),
        np.ones((r1, 1), dtype=int),
    )
    if i_forward is not None:
        indices = np.hstack(
            (np.kron(np.ones((s * r2, 1), dtype=int), i_forward), indices)
        )
    if i_backward is not None:
        indices = np.hstack(
            (indices, np.kron(i_backward, np.ones((r1 * s, 1), dtype=int)))
        )
    fiber = _evaluate(cross, indices)
    return fiber.reshape((r1, s, r2), order="F")


def _evaluate(cross: Cross, indices: np.ndarray) -> np.ndarray:
    """Evaluates the function at a tensor of indices."""
    if cross.structure == "binary":
        indices = _binary2decimal(
            indices, cross.qubits, cross.cross_strategy.mps_ordering
        )
    return np.array([cross.func(cross.mesh[idx]) for idx in indices])


# TODO: Clean and optimize
def _binary2decimal(
    indices: np.ndarray, qubits: List[int], mps_ordering: str
) -> np.ndarray:
    """Transforms a tensor of multi-indices in binary form to decimal form
    which can be used to evaluate function values. Follows the MPS ordering
    ("A" or "B") specified on the mps_ordering parameter of the strategy."""

    def bitlist_to_int(bitlist):
        out = 0
        for bit in bitlist:
            out = (out << 1) | bit
        return out

    m = len(qubits)
    decimal_indices = []
    for idx, n in enumerate(qubits):
        if mps_ordering == "A":
            rng = np.arange(idx * n, (idx + 1) * n)
        elif mps_ordering == "B":
            rng = np.arange(idx, m * n, m)
        else:
            raise ValueError("Invalid mps_ordering")
        decimal_ndx = bitlist_to_int(indices.T[rng])
        decimal_indices.append(decimal_ndx)
    decimal_indices = np.column_stack(decimal_indices)
    return decimal_indices


def _error_sampling(cross: Cross, sampling_points: int = 1000) -> float:
    """Returns the algorithm error given by comparing random samples with their exact values."""
    if cross.sweep == 1:
        cross.sampling_indices = np.vstack(
            [
                np.random.choice(k, sampling_points)
                for k in cross.mps.physical_dimensions()
            ]
        ).T
        cross.samples = _evaluate(cross, cross.sampling_indices)
    Q = cross.mps[0][0, cross.sampling_indices[:, 0], :]
    for i in range(1, cross.sites):
        Q = np.einsum("kq,qkr->kr", Q, cross.mps[i][:, cross.sampling_indices[:, i], :])
    error = np.linalg.norm(Q[:, 0] - cross.samples) / np.linalg.norm(cross.samples)
    return error


def _error_norm2(cross: Cross) -> float:
    """Returns the algorithm error given by evaluating the norm of its difference with respect to
    the previous sweep."""
    if cross.sweep == 1:
        cross.mps_prev = deepcopy(cross.mps)
        return 1
    error = abs(simplify(cross.mps - cross.mps_prev).norm())
    cross.mps_prev = cross.mps
    return error


def _converged(cross: Cross) -> bool:
    """Evaluates the convergence of the algorithm as defined by the strategy parameters."""
    return (
        cross.error < cross.cross_strategy.tol
        or cross.sweep >= cross.cross_strategy.maxiter
        or cross.maxrank >= cross.cross_strategy.maxrank
    )


def reorder_tensor(tensor: np.ndarray, qubits: List[int]) -> np.ndarray:
    """Reorders an A-ordered tensor into a B-ordered tensor (and the other way around)."""
    m = len(qubits)
    shape_orig = tensor.shape
    tensor = tensor.reshape([2] * sum(qubits))
    axes = [np.arange(idx, m * n, m) for idx, n in enumerate(qubits)]
    axes = [item for items in axes for item in items]
    tensor = np.transpose(tensor, axes=axes)
    return tensor.reshape(shape_orig)


def cross_interpolation(
    func: Callable,
    mesh: Mesh,
    mps: Optional[MPS] = None,
    cross_strategy: CrossStrategy = CrossStrategy(),
) -> MPS:
    """Tensor Cross Interpolation algorithm.

    This runs the Tensor Cross Interpolation algorithm in order to encode a black-box tensor
    given by a vector-valued function and a multidimensional mesh on a Matrix Product State (MPS).

    Parameters
    ----------
    func : Callable
        A multidimensional **vector-valued** function to be encoded in MPS form.
    mesh : Mesh
        A multidimensional discretized mesh on which the function is defined.
    mps : MPS
        An initial MPS with the same dimensions as the mesh to serve as an initial approximation.
    cross_strategy : CrossStrategy
        An object which contains the algorithm parameters.
    """
    if mps is None:
        if not all((s != 0) and (s & (s - 1) == 0) for s in mesh.shape()[:-1]):
            raise ValueError("The mesh size must be a power of two")
        sites = sum([int(np.log2(s)) for s in mesh.shape()[:-1]])
        mps = random_mps([2] * sites, 1, rng=np.random.default_rng(42))

    cross = Cross(func, mesh, mps, cross_strategy)
    _initialize(cross)

    while not _converged(cross):
        _sweep(cross)
        if cross_strategy.error_type == "sampling":
            error_name = "Sampling error"
            cross.error = _error_sampling(cross)
        elif cross_strategy.error_type == "norm":
            error_name = "Norm error"
            cross.error = _error_norm2(cross)
        else:
            raise ValueError("Invalid error_type")

        log(
            f"Sweep {cross.sweep:<3} | "
            + f"Max χ {cross.maxrank:>3} | "
            + f"{error_name} {cross.error:.2E}"
        )

    return cross.mps
