from copy import deepcopy
from dataclasses import dataclass
from time import perf_counter
from typing import Callable, Optional

import numpy as np
from scipy.linalg import lu, solve_triangular  # type: ignore

from ..state import MPS, Strategy, Truncation, Simplification
from ..tools import DEBUG, log
from ..truncate import SIMPLIFICATION_STRATEGY, simplify
from ..typing import *
from .integrals import integrate_mps
from .mesh import Mesh
from .sampling import random_mps_indices, sample_mps


DEFAULT_CROSS_STRATEGY = Strategy(
    method=Truncation.ABSOLUTE_SINGULAR_VALUE,
    tolerance=1e-8,
    simplify=Simplification.VARIATIONAL,
    simplification_tolerance=1e-8,
    normalize=False,
)


def maxvol_square(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns the optimal row indices and matrix of coefficients for the
    square maxvol decomposition of a given matrix.  This algorithm
    finds a submatrix of a 'tall' matrix with maximal volume
    (determinant of the square submatrix).

    Parameters
    ----------
    matrix : np.ndarray
        A 'tall' matrix (more rows than columns) for which the square
        maxvol decomposition is to be computed.

    Returns
    -------
    I : np.ndarray
        The optimal row indices of the tall matrix. These indices
        correspond to rows that form a square submatrix with maximal
        volume.
    B : np.ndarray
        The matrix of coefficients. This matrix represents the
        coefficients for the linear combination of rows in the
        original matrix that approximates the remaining rows, namely,
        a matrix B such that A ≈ B A[I, :].
    """
    SQUARE_MAXITER = 100
    SQUARE_TOL = 1.05
    n, r = matrix.shape
    if n <= r:
        raise ValueError('Input matrix should be "tall"')
    P, L, U = lu(matrix, check_finite=False)
    I = P[:, :r].argmax(axis=0)
    Q = solve_triangular(U, matrix.T, trans=1, check_finite=False)
    B = solve_triangular(
        L[:r, :], Q, trans=1, check_finite=False, unit_diagonal=True, lower=True
    ).T
    for _ in range(SQUARE_MAXITER):
        i, j = np.divmod(np.abs(B).argmax(), r)
        if np.abs(B[i, j]) <= SQUARE_TOL:
            break
        I[j] = i
        bj = B[:, j]
        bi = B[i, :].copy()
        bi[j] -= 1.0
        B -= np.outer(bj, bi / B[i, j])
    return I, B


def maxvol_rectangular(
    matrix: np.ndarray, min_rank_change: int, max_rank_change: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns the optimal row indices and matrix of coefficients for the
    maxvol algorithm applied to a tall matrix.  This algorithm extends
    the square maxvol algorithm to find a 'rectangular' submatrix with
    more rows than the columns of the original matrix.

    Parameters
    ----------
    matrix : np.ndarray
        A 'tall' matrix (more rows than columns) for which the
        rectangular maxvol decomposition is to be computed.
    min_rank_change : int
        The minimum number of rows to be added to the rank of the square submatrix.
    max_rank_change : int
        The maximum number of rows to be added to the rank of the square submatrix.

    Returns
    -------
    I : np.ndarray
        The optimal row indices of the tall matrix. These indices
        correspond to rows that form a rectangular submatrix with more
        rows than the columns of the original matrix.
    B : np.ndarray
        The matrix of coefficients. This matrix represents the
        coefficients of the linear combination of rows in the original
        matrix that approximates the remaining rows, namely, a matrix
        B such that A ≈ B A[I, :].
    """
    RECTANGULAR_TOL = 1.10
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
        if k >= r_min and F[i] <= RECTANGULAR_TOL**2:
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


def maxvol(matrix: np.ndarray, rank_change: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Chooses and applies the appropriate maxvol algorithm (square or
    rectangular) to find a submatrix with maximal volume, and returns
    the indices of these rows along with the matrix of coefficients.

    Parameters
    ----------
    matrix : np.ndarray
        The input matrix on which the maxvol algorithm is to be applied.
    rank_change : int
        The desired change in rank for the matrix, determining whether
        to use the square or rectangular maxvol algorithm.

    Returns
    -------
    indices : np.ndarray
        An array of indices corresponding to the rows that form a
        submatrix with maximal volume.
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


# TODO: We have to make I_s an ArrayLike because it is used as a
# list of as arrays later on. We should fix the TT Cross algorithm
# to avoid using lists at some point
def random_initial_indices(
    I_s: list[np.ndarray],
    starting_bond: int,
    rng: np.random.Generator = np.random.default_rng(),
) -> list[np.ndarray]:
    """
    Generates a list of random initial indices for tensor
    decomposition using a specified random number generator.  Each set
    of indices is chosen randomly from a given array of indices and
    progressively concatenated.

    Parameters
    ----------
    I_s : `numpy.typing.ArrayLike`
        An array of possible indices from which the random indices are
        to be selected (physical indices).
    starting_bond : int
        The number of indices to be chosen randomly at each step.
    rng : np.random.Generator, default=`numpy.random.default_rng()`
        The random number generator to be used. If None, uses Numpy's
        default random number generator without any predefined seed.

    Returns
    -------
    I_g : list[np.ndarray]
        A list of arrays, each containing a set of randomly chosen
        indices. The size of each array grows progressively as indices
        are concatenated at each step.
    """
    I_g = [np.array([], dtype=int)]
    for i, i_s in reversed(list(enumerate(I_s))):  # index counts in reverse order
        choice = rng.choice(i_s, size=starting_bond, replace=True).reshape(-1, 1)
        if i == len(I_s) - 1:
            I_g.insert(0, choice)
        else:
            I_g.insert(0, np.hstack([choice, I_g[0]]))
    return I_g


def sample_initial_indices(state: MPS) -> list[np.ndarray]:
    """
    Samples initial indices from a given MPS by performing a sweep of
    the cross interpolation algorithm.  This allows to give an
    arbitrary starting bond dimension and a hopefully better starting
    point leading to faster convergence.

    Parameters
    ----------
    state : MPS
        The MPS from which the initial indices are to be sampled.

    Returns
    -------
    I_g : list[np.ndarray]
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
        I_fiber_rows = np.hstack(
            (np.kron(_ones(r_g), I_s[k]), np.kron(I_g[k + 1], _ones(s)))
        )
        I_g[k] = I_fiber_rows[maxvol_rows, :]

    return I_g


def sample_tensor_fiber(
    func: Callable,
    mesh: Mesh,
    T: np.ndarray,
    i_le: np.ndarray,
    i_s: np.ndarray,
    i_g: np.ndarray,
) -> np.ndarray:
    """
    Samples the k-th tensor fiber of the underlying MPS representation
    of a function from a collection of multi-indices centered at a
    given site k. I.e, samples $A\left(J_{\le k-1}, i_k, J_{\g
    k}\right)$ (see dolgov2020 pag.7).

    Parameters
    ----------
    func : Callable
        The vector function to be evaluated.
    mesh : Mesh
        The mesh of points where the function is defined.
    mps_order : str
        The order of the MPS sites, determining with which
        transformation matrix the tensor fiber is sampled.
    i_le : np.ndarray
        The multi-indices coming from all sites smaller or equal to k
        ($J_{\le k-1}$) in the MPS.
    i_s : np.ndarray
        The physical indices, representing the physical dimension at
        site k of the underlying MPS.
    i_g : np.ndarray
        The multi-indices coming from all sites greater than k ($J_{\g
        k}$) in the MPS.

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


@dataclass
class CrossStrategy:
    maxbond: int = 100
    maxiter: int = 100
    tol: float = 1e-10
    bond_change: int = 1
    starting_indices: Optional[list[np.ndarray]] = None
    starting_bond: int = 1
    error_type: str = "sampling"
    mps_order: str = "A"
    """
    Configuration strategy for the tensor cross-interpolation algorithm.

    Parameters
    ----------
    maxbond : int, default=100
        Maximum bond dimension allowed for the MPS.
    maxiter : int, default=100
        Maximum number of sweeps allowed.
    tol : float, default=1e-10
        Error tolerance for convergence, of type as specified by error_type.
    bond_change : int, default=1
        Increment in bond dimension per half-sweep (twice this value
        per full left-right sweep).
    starting_indices : Optional[list[np.ndarray]], default=None
        Initial indices for the cross algorithm; if None, uses a
        number of starting_bond random indices.
    starting_bond : int, default=1
        Initial bond dimension for the MPS, used only when starting_indices is None.
    error_type : str, default="sampling"
        Method for error calculation; can be 'sampling', 'norm', or 'integral'.
    mps_order : str, default="A"
        Order of sites in MPS; can be 'A' or 'B', affecting the sampling strategy.
    """


def cross_interpolation(
    func: Callable,
    mesh: Mesh,
    cross_strategy: CrossStrategy = CrossStrategy(),
    strategy: Strategy = SIMPLIFICATION_STRATEGY.replace(normalize=False),
    callback: Optional[Callable] = None,
    rng: np.random.Generator = np.random.default_rng(),
) -> CrossResults:
    """
    Performs cross interpolation on a vectorized function discretized
    on a mesh to obtain its MPS representation using the maxvol
    algorithm on an Alternating Least Squares (ALS) scheme.

    Parameters
    ----------
    func : Callable
        The vectorized function to be approximated as a MPS.
    mesh : Mesh
        The mesh of points where the function is defined.
    cross_strategy : CrossStrategy, default=CrossStrategy()
        The strategy configuration for the algorithm.
    strategy : Strategy, default=Strategy()
        The MPS simplification strategy to perform at the end of the algorithm.
    callback : Optional[Callable], default=None
        An optional callback function called on the state after each sweep.

    Returns
    -------
    results: CrossResults
        The results of the cross interpolation process, including the
        final MPS state, convergence status, message, error, bond,
        time and evaluation count trajectories.
    """
    mps_indices = None
    mesh_samples = None
    data_prev = None
    integral_prev = None
    T = mesh.mps_to_mesh_matrix(cross_strategy.mps_order)

    def get_error(state: MPS):
        if cross_strategy.error_type == "sampling":
            nonlocal mps_indices, mesh_samples
            # TODO: Think about this and whether the state/sampling.py
            # module can be used instead.
            if sweep == 0:
                mps_indices = random_mps_indices(state, rng=rng)
                mesh_samples = func(mesh[mps_indices @ T]).reshape(-1)
            mps_samples = sample_mps(state, mps_indices)
            error = np.max(np.abs(mps_samples - mesh_samples))
        elif cross_strategy.error_type == "norm":
            nonlocal data_prev
            if sweep == 0:
                # Save a copy of the previous tensors
                data_prev = deepcopy(state._data)
                return np.Inf
            # TODO: Rethink about this way of checking convergence.
            # Right now it is computing the relative change in norm.
            # At the moment, this method is much more expensive than sampling.
            strategy = Strategy(normalize=False)
            state_prev = MPS(data_prev)
            error = abs(
                simplify(state - state_prev, strategy=strategy).norm()
                / state_prev.norm()
            )
            data_prev = deepcopy(state._data)
        elif cross_strategy.error_type == "integral":
            nonlocal integral_prev
            if sweep == 0:
                integral_prev = integrate_mps(state, mesh, integral_type="trapezoidal")
                return np.Inf
            # TODO: Implement Féjer quadrature (valid for arbitrary sites and exponentially convergent)
            integral = integrate_mps(state, mesh, integral_type="trapezoidal")
            if DEBUG:
                log(f"integral = {integral:.15e}")
            error = abs(integral - integral_prev)
            integral_prev = integral
        return error

    # TODO: Think how to generalize to arbitrary physical dimension
    base = 2
    mesh_sites_per_dimension = [int(np.emath.logn(base, s)) for s in mesh.dimensions]
    sites = sum(mesh_sites_per_dimension)
    I_s = [np.array([[0], [1]]) for _ in range(sites)]
    I_le = [np.array([], dtype=int)] * (sites + 1)
    if cross_strategy.starting_indices is None:
        I_g = random_initial_indices(
            I_s, starting_bond=cross_strategy.starting_bond, rng=rng
        )
    else:
        if len(cross_strategy.starting_indices) - 1 != sites:
            raise ValueError("Invalid starting indices")
        I_g = cross_strategy.starting_indices

    errors = []
    bonds = []
    evaluations = []
    times = []

    state = MPS([np.zeros((1, 2, 1))] * sites)  # Placeholder state

    converged = False
    message = f"Maximum number of sweeps {cross_strategy.maxiter} reached"
    _ones = lambda x: np.ones((x, 1), dtype=int)

    for sweep in range(cross_strategy.maxiter):
        start_time = perf_counter()
        # Update maxvol rank change based on maxbond
        if max(A.shape[0] for A in state) < cross_strategy.maxbond:
            rank_change = cross_strategy.bond_change
        else:
            rank_change = 0
        evals = 0

        # Forward pass
        for k in range(sites):
            fiber = sample_tensor_fiber(func, mesh, T, I_le[k], I_s[k], I_g[k + 1])
            evals += fiber.size
            r_le, s, r_g = fiber.shape
            fiber_matrix = fiber.reshape(r_le * s, r_g, order="F")
            Q, R = np.linalg.qr(fiber_matrix)
            maxvol_rows, maxvol_coefs = maxvol(Q, rank_change)
            state[k] = maxvol_coefs.reshape(r_le, s, -1, order="F")
            I_fiber_rows = np.hstack(
                (np.kron(_ones(s), I_le[k]), np.kron(I_s[k], _ones(r_le)))
            )
            I_le[k + 1] = I_fiber_rows[maxvol_rows, :]
        state[k] = np.tensordot(state[k], Q[maxvol_rows, :] @ R, 1)

        # Backward pass
        for k in reversed(range(sites)):
            fiber = sample_tensor_fiber(func, mesh, T, I_le[k], I_s[k], I_g[k + 1])
            evals += fiber.size
            r_le, s, r_g = fiber.shape
            fiber_matrix = fiber.reshape(r_le, s * r_g, order="F").T
            Q, R = np.linalg.qr(fiber_matrix)
            maxvol_rows, maxvol_coefs = maxvol(Q, rank_change)
            state[k] = maxvol_coefs.T.reshape(-1, s, r_g, order="F")
            I_fiber_rows = np.hstack(
                (np.kron(_ones(r_g), I_s[k]), np.kron(I_g[k + 1], _ones(s)))
            )
            I_g[k] = I_fiber_rows[maxvol_rows, :]
        state[0] = np.tensordot((Q[maxvol_rows, :] @ R).T, state[0], 1)

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

        if error < cross_strategy.tol:
            message = f"State converged within tolerance {cross_strategy.tol}"
            converged = True
            break

        if callback is not None:
            callback(state)

    # Simplify the state according to the strategy
    state = simplify(state, strategy=strategy)
    truncated_bonds = state.bond_dimensions()
    truncated_error = get_error(state)
    bonds.append(truncated_bonds)
    errors.append(truncated_error)
    if DEBUG:
        log(
            f"simplification truncates maxbond from {max(all_bonds)} to {max(truncated_bonds)} with error {truncated_error:.15e}"
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
