from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from copy import copy
from typing import Callable, Optional
from .maxvol import maxvol_sqr, maxvol_rct
from ..analysis.mesh import Mesh
from ..tools import log, DEBUG
from ..state import MPS, random_mps
from ..truncate import simplify


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
    sites: int
    qubits: list[int]
    sweeps: int
    error: float
    I_physical: list[np.ndarray]
    I_forward: list[Optional[np.ndarray]]
    I_backward: list[Optional[np.ndarray]]

    def __init__(
        self, func: Callable, mesh: Mesh, mps: MPS, cross_strategy: CrossStrategy
    ):
        self.func = func
        self.mesh = mesh
        self.mps = mps
        self.cross_strategy = cross_strategy

        mps_shape = tuple(self.mps.physical_dimensions())
        mesh_shape = self.mesh.shape()[:-1]
        if np.prod(mps_shape) == np.prod(mesh_shape) and all(
            dim == 2 for dim in mps_shape
        ):
            self.structure = "binary"
        elif mps_shape == mesh_shape:
            self.structure = "tt"
        else:
            raise ValueError("Non-matching mesh and initial MPS")
        self.sites = len(self.mps)
        self.qubits = [int(np.log2(s)) for s in mesh_shape]
        self.I_physical = [
            np.arange(k, dtype=int).reshape(-1, 1)
            for k in self.mps.physical_dimensions()
        ]
        self.I_forward = [None for _ in range(self.sites + 1)]
        self.I_backward = [None for _ in range(self.sites + 1)]
        self.error = 1
        self.sweeps = 0

    def presweep(cross: Cross) -> None:
        """Executes a presweep on the initial MPS without evaluating
        the function and using the square maxvol algorithm without rank
        increments."""
        cross_initial = copy(cross)
        cross_initial.cross_strategy = CrossStrategy(
            maxvol_rct_minrank=0, maxvol_rct_maxrank=0
        )

        # Forward pass
        R = np.ones((1, 1))
        for j in range(cross.sites):
            fiber = np.tensordot(R, cross.mps[j], 1)
            cross.mps[j], cross.I_forward[j + 1], R = cross_initial.skeleton(
                fiber, j, ltr=True
            )
        cross.mps[cross.sites - 1] = np.tensordot(cross.mps[cross.sites - 1], R, 1)

        # Backward pass
        R = np.ones((1, 1))
        for j in reversed(range(cross.sites)):
            fiber = np.tensordot(cross.mps[j], R, 1)
            cross.mps[j], cross.I_backward[j], R = cross_initial.skeleton(
                fiber, j, ltr=False
            )
        cross.mps[0] = np.tensordot(R, cross.mps[0], 1)

    def sweep(self: Cross) -> None:
        """Runs a forward-backward sweep on the MPS that iteratively updates
        its tensors and forward / backward multi-indices (pivots) by means
        of the skeleton decomposition."""
        # Forward pass
        R = np.ones((1, 1))
        for j in range(self.sites):
            fiber = self.sample(j)
            self.mps[j], self.I_forward[j + 1], R = self.skeleton(fiber, j, ltr=True)
        self.mps[self.sites - 1] = np.tensordot(self.mps[self.sites - 1], R, 1)

        # Backward pass
        R = np.ones((1, 1))
        for j in reversed(range(self.sites)):
            fiber = self.sample(j)
            self.mps[j], self.I_backward[j], R = self.skeleton(fiber, j, ltr=False)
        self.mps[0] = np.tensordot(R, self.mps[0], 1)

        self.sweeps += 1

    def maximum_bond_dimension(self: Cross) -> int:
        """Return the maximum bond dimension reached"""
        return max(A.shape[0] for A in self.mps)

    # TODO: Clean and optimize
    def skeleton(
        self: Cross,
        fiber: np.ndarray,
        j: int,
        ltr: bool,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns the skeleton decomposition of a tensor fiber using either
        the square or rectangular (rank-increasing) maxvol algorithm.
        """
        r1, s, r2 = fiber.shape

        # Reshape the fiber into a matrix
        if ltr:
            i_virtual = self.I_forward[j]
            fiber = np.reshape(fiber, (r1 * s, r2), order="F")
        else:
            i_virtual = self.I_backward[j + 1]
            fiber = np.reshape(fiber, (r1, s * r2), order="F").T

        # Perform QR factorization
        Q, R = np.linalg.qr(fiber)

        k = self.cross_strategy.maxvol_sqr_maxiter
        e = self.cross_strategy.maxvol_sqr_tau
        tau = self.cross_strategy.maxvol_rct_tau
        min_r = self.cross_strategy.maxvol_rct_minrank
        max_r = self.cross_strategy.maxvol_rct_maxrank
        if max(self.mps.bond_dimensions()) >= self.cross_strategy.maxrank:
            min_r = 0
            max_r = 0

        # Perform maxvol decomposition on the Q-factor
        if Q.shape[0] <= Q.shape[1]:
            i_maxvol = np.arange(Q.shape[0], dtype=int)
            Q_maxvol = np.eye(Q.shape[0])
        elif self.cross_strategy.maxvol_rct_maxrank == 0:
            i_maxvol, Q_maxvol = maxvol_sqr(Q, k, e)
        else:
            i_maxvol, Q_maxvol = maxvol_rct(Q, k, e, tau, min_r, max_r)

        # Redefine the fiber as the decomposed Q-factor
        i_physical = self.I_physical[j]
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

    def sample(self: Cross, j: int) -> np.ndarray:
        """Returns a fiber of the implicit tensor along the site j by evaluating
        the function at the kronecker product of the physical indices and the
        forward and backward maxvol multi-indices (pivots) that are present
        at that site.
        """
        i_physical = self.I_physical[j]
        i_forward = self.I_forward[j]
        i_backward = self.I_backward[j + 1]
        r1 = 1 if i_forward is None else i_forward.shape[0]
        s = i_physical.shape[0]
        r2 = 1 if i_backward is None else i_backward.shape[0]
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
        fiber = self.evaluate(indices)
        return fiber.reshape((r1, s, r2), order="F")

    def evaluate(self: Cross, indices: np.ndarray) -> np.ndarray:
        """Evaluates the function at a tensor of indices."""
        if self.structure == "binary":
            indices = _binary2decimal(
                indices, self.qubits, self.cross_strategy.mps_ordering
            )
        return np.array([self.func(self.mesh[idx]) for idx in indices]).reshape(-1)

    def sampling_error(self: Cross, sampling_points: int = 1000) -> float:
        """Returns the algorithm error given by comparing random samples
        with their exact values."""
        if self.sweeps == 1:
            self.sampling_indices = np.vstack(
                [
                    np.random.choice(k, sampling_points)
                    for k in self.mps.physical_dimensions()
                ]
            ).T
            self.samples = self.evaluate(self.sampling_indices)
        Q = self.mps[0][0, self.sampling_indices[:, 0], :]
        for i in range(1, self.sites):
            Q = np.einsum(
                "kq,qkr->kr", Q, self.mps[i][:, self.sampling_indices[:, i], :]
            )
        error = np.linalg.norm(Q[:, 0] - self.samples) / np.linalg.norm(self.samples)
        return error  # type: ignore

    def norm2_error(self: Cross) -> float:
        """Returns the algorithm error given by evaluating the norm of
        its difference with respect to the previous sweep."""
        if self.sweeps == 1:
            self.mps_prev = self.mps
            return np.Inf
        error = abs(simplify(self.mps - self.mps_prev).norm())
        self.mps_prev = self.mps
        return error

    def converged(self: Cross) -> bool:
        """Evaluates the convergence of the algorithm as defined by the
        strategy parameters."""
        return (
            self.error < self.cross_strategy.tol
            or self.sweeps >= self.cross_strategy.maxiter
        )


# TODO: Clean and optimize
def _binary2decimal(
    indices: np.ndarray, qubits: list[int], mps_ordering: str
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
    return np.column_stack(decimal_indices)


def reorder_tensor(tensor: np.ndarray, qubits: list[int]) -> np.ndarray:
    """Reorders an A-ordered tensor into a B-ordered tensor (and the other
    way around)."""
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

    This runs the Tensor Cross Interpolation algorithm in order to encode
    a black-box tensor given by a vector-valued function and a multidimensional
    mesh on a Matrix Product State (MPS).

    Parameters
    ----------
    func : Callable
        A multidimensional **vector-valued** function to be encoded in MPS form.
    mesh : Mesh
        A multidimensional discretized mesh on which the function is defined.
    mps : MPS
        An initial MPS with the same dimensions as the mesh to serve as an
        initial approximation.
    cross_strategy : CrossStrategy
        An object which contains the algorithm parameters.
    """
    if mps is None:
        if not all((s != 0) and (s & (s - 1) == 0) for s in mesh.shape()[:-1]):
            raise ValueError("The mesh size must be a power of two")
        sites = sum([int(np.log2(s)) for s in mesh.shape()[:-1]])
        mps = random_mps([2] * sites, 1, rng=np.random.default_rng(42))

    cross = Cross(func, mesh, mps, cross_strategy)
    cross.presweep()

    while not cross.converged():
        cross.sweep()
        if cross_strategy.error_type == "sampling":
            error_name = "Sampling error"
            cross.error = cross.sampling_error()
        elif cross_strategy.error_type == "norm":
            error_name = "Norm error"
            cross.error = cross.norm2_error()
        else:
            raise ValueError("Invalid error_type")
        if DEBUG:
            log(
                f"Sweep {cross.sweeps:<3} | "
                + f"Max χ {cross.maximum_bond_dimension():>3} | "
                + f"{error_name} {cross.error:.2E}"
            )

    return cross.mps
