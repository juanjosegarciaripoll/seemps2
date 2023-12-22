from copy import deepcopy
from dataclasses import dataclass
from time import perf_counter
from typing import Callable, List, Optional, Tuple

import numpy as np
from scipy.linalg import lu, solve_triangular  # type: ignore

from ..state import MPS, Strategy
from ..tools import DEBUG, log
from ..truncate import SIMPLIFICATION_STRATEGY, simplify
from ..typing import *
from .integrals import integrate_mps
from .mesh import Mesh
from .sampling import random_mps_indices, sample_mps


def maxvol_square(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns the optimal row indices and matrix of coefficients for the square maxvol decomposition of a given matrix.
    This algorithm finds a submatrix of a 'tall' matrix with maximal volume (determinant of the square submatrix).

    Parameters
    ----------
    matrix : np.ndarray
        A 'tall' matrix (more rows than columns) for which the square maxvol decomposition is to be computed.

    Returns
    -------
    I : np.ndarray
        The optimal row indices of the tall matrix. These indices correspond to rows that form a square submatrix
        with maximal volume.
    B : np.ndarray
        The matrix of coefficients. This matrix represents the coefficients for the linear combination of rows in the
        original matrix that approximates the remaining rows, namely, a matrix B such that A ≈ B A[I, :].
    """
    square_maxiter = 100
    square_tol = 1.05
    n, r = matrix.shape
    if n <= r:
        raise ValueError('Input matrix should be "tall"')
    P, L, U = lu(matrix, check_finite=False)
    I = P[:, :r].argmax(axis=0)
    Q = solve_triangular(U, matrix.T, trans=1, check_finite=False)
    B = solve_triangular(
        L[:r, :], Q, trans=1, check_finite=False, unit_diagonal=True, lower=True
    ).T
    for _ in range(square_maxiter):
        i, j = np.divmod(np.abs(B).argmax(), r)
        if np.abs(B[i, j]) <= square_tol:
            break
        I[j] = i
        bj = B[:, j]
        bi = B[i, :].copy()
        bi[j] -= 1.0
        B -= np.outer(bj, bi / B[i, j])
    return I, B


def maxvol_rectangular(
    matrix: np.ndarray, min_rank_change: int, max_rank_change: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns the optimal row indices and matrix of coefficients for the maxvol algorithm applied to a tall matrix.
    This algorithm extends the square maxvol algorithm to find a 'rectangular' submatrix with more rows than the columns
    of the original matrix.

    Parameters
    ----------
    matrix : np.ndarray
        A 'tall' matrix (more rows than columns) for which the rectangular maxvol decomposition is to be computed.
    min_rank_change : int
        The minimum number of rows to be added to the rank of the square submatrix.
    max_rank_change : int
        The maximum number of rows to be added to the rank of the square submatrix.

    Returns
    -------
    I : np.ndarray
        The optimal row indices of the tall matrix. These indices correspond to rows that form a rectangular
        submatrix with more rows than the columns of the original matrix.
    B : np.ndarray
        The matrix of coefficients. This matrix represents the coefficients of the linear combination of rows
        in the original matrix that approximates the remaining rows, namely, a matrix B such that A ≈ B A[I, :].
    """
    rectangular_tol = 1.10
    n, r = matrix.shape
    r_min = r + min_rank_change
    r_max = r + max_rank_change if max_rank_change is not None else n
    r_max = min(r_max, n)
    if r_min < r or r_min > r_max or r_max > n:
        raise ValueError("Invalid minimum/maximum number of added rows")
    I0, B = maxvol_square(matrix)
    I = np.hstack([I0, np.zeros(r_max - r, dtype=I0.dtype)])
    S = np.ones(n, dtype=int)
    S[I0] = 0
    F = S * np.linalg.norm(B, axis=1) ** 2
    for k in range(r, r_max):
        i = np.argmax(F)
        if k >= r_min and F[i] <= rectangular_tol**2:
            break
        I[k] = i
        S[i] = 0
        v = B.dot(B[i])
        l = 1.0 / (1 + v[i])
        B = np.hstack([B - l * np.outer(v, B[i]), l * v.reshape(-1, 1)])
        F = S * (F - l * v * v)
    I = I[: B.shape[1]]
    B[I] = np.eye(B.shape[1], dtype=B.dtype)
    return I, B


def maxvol(matrix: np.ndarray, rank_change: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Chooses and applies the appropriate maxvol algorithm (square or rectangular) to find a submatrix
    with maximal volume, and returns the indices of these rows along with the matrix of coefficients.

    Parameters
    ----------
    matrix : np.ndarray
        The input matrix on which the maxvol algorithm is to be applied.
    rank_change : int
        The desired change in rank for the matrix, determining whether to use the square or rectangular
        maxvol algorithm.

    Returns
    -------
    indices : np.ndarray
        An array of indices corresponding to the rows that form a submatrix with maximal volume.
    coefficients : np.ndarray
        A matrix of coefficients such that A ≈ B A[I, :].
    """

    n, r = matrix.shape
    max_rank_change = min(rank_change, n - r)
    min_rank_change = min(rank_change, max_rank_change)
    if n <= r:
        indices = np.arange(n, dtype=int)
        coefficients = np.eye(n)
    elif max_rank_change == 0:
        indices, coefficients = maxvol_square(matrix)
    else:
        indices, coefficients = maxvol_rectangular(
            matrix, min_rank_change, max_rank_change
        )

    return indices, coefficients


def random_initial_indices(
    I_s: np.ndarray,
    starting_bond: int,
    rng: Optional[np.random.Generator] = None,
) -> List[np.ndarray]:
    """
    Generates a list of random initial indices for tensor decomposition using a specified random number generator.
    Each set of indices is chosen randomly from a given array of indices and progressively concatenated.

    Parameters
    ----------
    I_s : np.ndarray
        An array of possible indices from which the random indices are to be selected (physical indices).
    starting_bond : int
        The number of indices to be chosen randomly at each step.
    rng : Optional[np.random.Generator], default=None
        The random number generator to be used. If None, a default generator with a fixed seed is used.

    Returns
    -------
    I_g : List[np.ndarray]
        A list of arrays, each containing a set of randomly chosen indices. The size of each array grows progressively
        as indices are concatenated at each step.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    I_g = [np.array([], dtype=int)]
    for i, i_s in reversed(list(enumerate(I_s))):
        choice = rng.choice(i_s, size=starting_bond, replace=True).reshape(-1, 1)
        if i == len(I_s) - 1:
            I_g.insert(0, choice)
        else:
            I_g.insert(0, np.hstack([choice, I_g[0]]))
    return I_g


def sample_initial_indices(state: MPS) -> List[np.ndarray]:
    """
    Samples initial indices from a given MPS by performing a sweep of the cross interpolation algorithm.
    This allows to give an arbitrary starting bond dimension and a hopefully better starting point leading
    to faster convergence.

    Parameters
    ----------
    state : MPS
        The MPS from which the initial indices are to be sampled.

    Returns
    -------
    I_g : List[np.ndarray]
        A list of arrays containing the sampled indices.
    """
    rank_change = 0
    sites = len(state)
    I_s = [np.array([[0], [1]]) for _ in range(sites)]
    I_le = [np.array([], dtype=int)] * (sites + 1)
    I_g = [np.array([], dtype=int)] * (sites + 1)
    _ones = lambda x: np.ones((x, 1), dtype=int)

    # Forward pass
    R = np.ones((1, 1))
    for k in range(sites):
        fiber = np.tensordot(R, state[k], 1)
        r_le, s, r_g = fiber.shape
        fiber_matrix = fiber.reshape(r_le * s, r_g)
        Q, R = np.linalg.qr(fiber_matrix)
        maxvol_rows, _ = maxvol(Q, rank_change)
        R = Q[maxvol_rows, :] @ R
        I_fiber_rows = np.hstack(
            (np.kron(_ones(s), I_le[k]), np.kron(I_s[k], _ones(r_le)))
        )
        I_le[k + 1] = I_fiber_rows[maxvol_rows, :]

    # Backward pass
    R = np.ones((1, 1))
    for k in reversed(range(sites)):
        fiber = np.tensordot(state[k], R, 1)
        r_le, s, r_g = fiber.shape
        fiber_matrix = fiber.reshape(r_le, s * r_g)
        Q, R = np.linalg.qr(fiber_matrix.T)
        maxvol_rows, _ = maxvol(Q, rank_change)
        R = (Q[maxvol_rows, :] @ R).T
        I_fiber_cols = np.hstack(
            (np.kron(_ones(r_g), I_s[k]), np.kron(I_g[k + 1], _ones(s)))
        )
        I_g[k] = I_fiber_cols[maxvol_rows, :]

    return I_g


def sample_tensor_fiber(
    func: Callable,
    mesh: Mesh,
    mps_ordering: str,
    i_le: np.ndarray,
    i_s: np.ndarray,
    i_g: np.ndarray,
) -> np.ndarray:
    """
    Samples the k-th tensor fiber of the underlying MPS representation of a function from a collection of multi-indices
    centered at a given site k. I.e, samples $A\left(J_{\le k-1}, i_k, J_{\g k}\right)$ (see dolgov2020 pag.7).

    Parameters
    ----------
    func : Callable
        The vector function to be evaluated.
    mesh : Mesh
        The mesh of points where the function is defined.
    mps_ordering : str
        The ordering of the MPS sites, determining with which transformation matrix the tensor fiber is sampled.
    i_le : np.ndarray
        The multi-indices coming from all sites smaller or equal to k ($J_{\le k-1}$) in the MPS.
    i_s : np.ndarray
        The physical indices, representing the physical dimension at site k of the underlying MPS.
    i_g : np.ndarray
        The multi-indices coming from all sites greater than k ($J_{\g k}$) in the MPS.

    Returns
    -------
    fiber : np.ndarray
        The sampled tensor fiber.
    """
    _ones = lambda x: np.ones((x, 1), dtype=int)
    r_le = i_le.shape[0] if i_le.size > 0 else 1
    r_g = i_g.shape[0] if i_g.size > 0 else 1
    s = i_s.shape[0]
    # Extend the i_s to allow the left and right multi-indices to get attached.
    mps_indices = np.kron(np.kron(_ones(r_g), i_s), _ones(r_le))
    # Attach the multi-indices to the left and right of the extended i_s.
    if i_le.size > 0:
        mps_indices = np.hstack((np.kron(_ones(s * r_g), i_le), mps_indices))
    if i_g.size > 0:
        mps_indices = np.hstack((mps_indices, np.kron(i_g, _ones(r_le * s))))
    # Evaluate the function at the fiber indices
    T = mesh.binary_transformation_matrix(mps_ordering)
    fiber = func(mesh[mps_indices @ T]).reshape((r_le, s, r_g), order="F")
    return fiber


@dataclass
class CrossResults:
    state: MPS
    converged: bool
    message: str
    errors: VectorLike
    bonds: VectorLike
    evaluations: VectorLike
    times: VectorLike


def cross_interpolation(
    func: Callable,
    mesh: Mesh,
    starting_indices: Optional[List[np.ndarray]] = None,
    bond_change: int = 1,
    maxbond: int = 100,
    maxiter: int = 100,
    tol: float = 1e-10,
    error_type: str = "sampling",
    mps_ordering: str = "A",
    strategy: Strategy = SIMPLIFICATION_STRATEGY.replace(normalize=False),
    callback: Optional[Callable] = None,
) -> CrossResults:
    """
    Performs cross interpolation on a vectorized function discretized on a mesh to obtain its
    MPS representation using the maxvol algorithm on an Alternating Least Squares (ALS) scheme.

    Parameters
    ----------
    func : Callable
        The vectorized function to be approximated as a MPS.
    mesh : Mesh
        The mesh of points where the function is defined.
    starting_indices : Optional[List[np.ndarray]], default=None
        The initial indices I_g for the algorithm. If None, random indices with bond dimension 1 are used.
    bond_change : int, default=1
        The increment in the bond dimension at each sweep until reaching maxbond.
    maxbond : int, default=100
        The maximum bond dimension allowed for the MPS.
    maxiter : int, default=100
        The maximum number of sweeps allowed.
    tol : float, default=1e-10
        The error tolerance for convergence.
    mps_ordering : str, default="A"
        Determines the order of the sites in the MPS by changing how the function is sampled.
    strategy : Strategy, default=Strategy()
        The MPS simplification strategy to perform at the end of the algorithm.
    callback : Optional[Callable], default=None
        An optional callback function called on the state after each sweep.

    Returns
    -------
    CrossResults
        The results of the cross interpolation process, including the final MPS state, convergence status,
        message, error, bond, time and evaluation count trajectories.
    """

    def get_error(state: MPS):
        if error_type == "sampling":
            mps_indices = random_mps_indices(state)
            T = mesh.binary_transformation_matrix(mps_ordering)
            mesh_samples = func(mesh[mps_indices @ T])
            mps_samples = sample_mps(state, mps_indices)
            error = np.max(np.abs(mps_samples - mesh_samples))
        elif error_type == "norm":
            if sweep == 0:
                # Save a deepcopy (needs to copy the previous tensors) to a function attribute
                get_error.state_prev = deepcopy(state)
                return np.Inf
            # TODO: Rethink about this way of checking convergence.
            # Right now it is computing the relative change in norm.
            # At the moment, this method is much more expensive than sampling.
            strategy = Strategy(normalize=False)
            error = abs(
                simplify(state - get_error.state_prev, strategy=strategy).norm()
                / get_error.state_prev.norm()
            )
            get_error.state_prev = deepcopy(state)
        elif error_type == "integral":
            if sweep == 0:
                get_error.integral_prev = integrate_mps(
                    state, mesh, integral_type="trapezoidal"
                )
                return np.Inf
            # TODO: Implement Féjer quadrature (valid for arbitrary sites and exponentially convergent)
            integral = integrate_mps(state, mesh, integral_type="trapezoidal")
            if DEBUG:
                log(f"integral = {integral:.15e}")
            error = abs(integral - get_error.integral_prev)
            get_error.integral_prev = integral
        return error

    mesh_shape = mesh.shape()[:-1]
    # TODO: Think how to generalize to arbitrary physical dimension
    base = 2
    mesh_sites_per_dimension = [int(np.emath.logn(base, s)) for s in mesh_shape]
    sites = sum(mesh_sites_per_dimension)
    I_s = [np.array([[0], [1]]) for _ in range(sites)]
    I_le = [np.array([], dtype=int)] * (sites + 1)
    if starting_indices is None:
        rng = np.random.default_rng(seed=42)
        I_g = random_initial_indices(I_s, starting_bond=1, rng=rng)
    else:
        if len(starting_indices) - 1 != sites:
            raise ValueError("Invalid starting indices")
        I_g = starting_indices

    errors = []
    bonds = []
    evaluations = []
    times = []

    state = MPS([np.zeros((1, 2, 1))] * sites)  # Placeholder state

    converged = False
    message = f"Maximum number of sweeps {maxiter} reached"
    _ones = lambda x: np.ones((x, 1), dtype=int)

    for sweep in range(maxiter):
        start_time = perf_counter()
        # Update maxvol rank change based on maxbond
        if max(A.shape[0] for A in state) < maxbond:
            rank_change = bond_change
        else:
            rank_change = 0
        evals = 0

        # Forward pass
        for k in range(sites):
            fiber = sample_tensor_fiber(
                func, mesh, mps_ordering, I_le[k], I_s[k], I_g[k + 1]
            )
            evals += fiber.size
            r_le, s, r_g = fiber.shape
            fiber_matrix = fiber.reshape(r_le * s, r_g, order="F")
            Q, R = np.linalg.qr(fiber_matrix)
            maxvol_rows, maxvol_coefs = maxvol(Q, rank_change)
            state[k] = maxvol_coefs.reshape(r_le, s, -1, order="F")
            R = Q[maxvol_rows, :] @ R
            I_fiber_rows = np.hstack(
                (np.kron(_ones(s), I_le[k]), np.kron(I_s[k], _ones(r_le)))
            )
            I_le[k + 1] = I_fiber_rows[maxvol_rows, :]
        state[k] = np.tensordot(state[k], R, 1)

        # Backward pass
        for k in reversed(range(sites)):
            fiber = sample_tensor_fiber(
                func, mesh, mps_ordering, I_le[k], I_s[k], I_g[k + 1]
            )
            evals += fiber.size
            r_le, s, r_g = fiber.shape
            fiber_matrix = fiber.reshape(r_le, s * r_g, order="F").T
            Q, R = np.linalg.qr(fiber_matrix)
            maxvol_rows, maxvol_coefs = maxvol(Q, rank_change)
            state[k] = maxvol_coefs.T.reshape(-1, s, r_g, order="F")
            R = (Q[maxvol_rows, :] @ R).T
            I_fiber_rows = np.hstack(
                (np.kron(_ones(r_g), I_s[k]), np.kron(I_g[k + 1], _ones(s)))
            )
            I_g[k] = I_fiber_rows[maxvol_rows, :]
        state[0] = np.tensordot(R, state[0], 1)

        end_time = perf_counter()
        time = end_time - start_time
        error = get_error(state)
        all_bonds = state.bond_dimensions()
        if DEBUG:
            log(
                f"sweep = {sweep:3d}, error = {error:.15e}, maxbond = {max(all_bonds):3d}, evaluations = {evals:8d}, time = {time:5f}"
            )
        errors.append(error)
        bonds.append(all_bonds)
        evaluations.append(evals)
        times.append(time)

        if error < tol:
            message = f"State converged within tolerance {tol}"
            converged = True
            break

        if callback is not None:
            callback(state)

    # Simplify the state according to the strategy
    state = simplify(state, strategy=strategy)
    truncated_bonds = state.bond_dimensions()
    bonds.append(truncated_bonds)
    if DEBUG:
        log(
            f"simplification truncates maxbond from {max(all_bonds)} to {max(truncated_bonds)}"
        )

    return CrossResults(
        state=state,
        converged=converged,
        message=message,
        errors=errors,
        bonds=bonds,
        evaluations=evaluations,
        times=times,
    )
