from __future__ import annotations
import numpy as np
import scipy
import dataclasses
import functools
from typing import Callable, TypeAlias

from ...state import MPS
from ...tools import make_logger
from ...typing import Vector, Matrix, Tensor3, Tensor4
from ..evaluation import random_mps_indices, evaluate_mps
from .black_box import BlackBox


def cross_interpolation(
    black_box: BlackBox,
    cross_strategy: CrossStrategy,
    initial_points: Matrix | None = None,
) -> CrossResults:
    """
    Computes the MPS representation of a black box function using some TCI variant.
    The TCI variant is dynamically dispatched based on the type of the given CrossStrategy.

    Parameters
    ----------
    black_box : BlackBox
        Black box to approximate as a MPS.
    cross_strategy : CrossStrategy
        Dataclass containing the parameters of the algorithm.
    initial_points : Optional[Matrix], default=None
        Coordinates of initial discretization points used to initialize the algorithm.
        Defaults to the origin.

    Returns
    -------
    CrossResults
        Dataclass containing the MPS representation of the black-box function,
        among other useful information.
    """
    return cross_strategy.algorithm(black_box, cross_strategy, initial_points)


@dataclasses.dataclass
class CrossStrategy:
    tol: float = 1e-8
    num_samples: int = 2**10
    error_norm: float = np.inf
    error_relative: bool = False
    max_iters: int = 200
    max_bond: int = 1000
    max_time: float | None = None
    max_evals: int | None = None
    rng: np.random.Generator = dataclasses.field(
        default_factory=lambda: np.random.default_rng()
    )
    """
    Dataclass containing the base parameters for TCI.

    Parameters
    ----------
    tol : float, default=1e-12
        Tolerance for the sampled error.
    num_samples : int, default=1024
        Number of function samples to evaluate the error.
    error_norm : float, default=np.inf
        L^p norm used for computing the error.
    error_relative : bool, default=False
        Whether to compute the absolute or relative error.
    max_iters : int, default=200
        Maximum number of iterations (half-sweeps) allowed.
    max_bond : int, default=1000
        Maximum MPS bond dimension allowed.
    max_time : Optional[float], default=None
        Maximum computation time allowed.
    max_evals : Optional[int], default=None
        Maximum number of evaluations allowed.
    rng : np.random.Generator, default=np.random.default_rng()
        Random number generator used to initialize the algorithm and sample the error.
    """

    def __post_init__(self) -> None:
        assert self.max_iters > 0
        assert self.num_samples > 0

    @property
    def algorithm(self) -> Callable:
        raise NotImplementedError("Subclasses must override 'algorithm'")


IndexMatrix: TypeAlias = np.ndarray[tuple[int, int], np.dtype[np.integer]]
IndexVector: TypeAlias = np.ndarray[tuple[int], np.dtype[np.integer]]
IndexSlice: TypeAlias = np.intp | IndexVector | slice


class CrossInterpolation:
    """
    Auxiliar base class for TCI used to keep track of the required interpolation
    information.
    """

    black_box: BlackBox
    sites: int
    I_l: list[np.ndarray]
    I_g: list[np.ndarray]
    I_s: list[np.ndarray]
    mps: MPS

    def __init__(self, black_box: BlackBox, initial_points: Matrix | None = None):
        self.black_box = black_box
        self.sites = len(black_box.physical_dimensions)
        if initial_points is None:
            initial_points = np.zeros(self.sites, dtype=int)
        self.I_l, self.I_g = self.points_to_indices(initial_points)
        self.I_s = [np.arange(s).reshape(-1, 1) for s in black_box.physical_dimensions]
        self.mps = MPS([np.ones((1, s, 1)) for s in black_box.physical_dimensions])

    def sample_fiber(self, k: int) -> Tensor3:
        i_l, i_s, i_g = self.I_l[k], self.I_s[k], self.I_g[k]
        mps_indices = self.combine_indices(i_l, i_s, i_g)
        return self.black_box[mps_indices].reshape((len(i_l), len(i_s), len(i_g)))

    def sample_superblock(self, k: int) -> Tensor4:
        i_l, i_g = self.I_l[k], self.I_g[k + 1]
        i_s1, i_s2 = self.I_s[k], self.I_s[k + 1]
        mps_indices = self.combine_indices(i_l, i_s1, i_s2, i_g)
        return self.black_box[mps_indices].reshape(
            (len(i_l), len(i_s1), len(i_s2), len(i_g))
        )

    @staticmethod
    def combine_indices(*indices: IndexMatrix, row_major: bool = False) -> IndexMatrix:
        """
        Computes the Cartesian product of a set of multi-indices arrays and arranges the
        result as concatenated indices in column or row-major order.

        Parameters
        ----------
        indices : *np.ndarray
            A variable number of arrays where each array is treated as a set of multi-indices.
        row_major : bool, default=False
            Whether to compute the Cartesian product in row-major order.

        Examples
        --------
        >>> combine_indices(np.array([[1, 2, 3], [4, 5, 6]]), np.array([[0], [1]]))
        array([[1, 2, 3, 0],
            [1, 2, 3, 1],
            [4, 5, 6, 0],
            [4, 5, 6, 1]])
        """

        def cartesian_column(A: Matrix, B: Matrix) -> Matrix:
            A_repeated = np.repeat(A, repeats=B.shape[0], axis=0)
            B_tiled = np.tile(B, (A.shape[0], 1))
            return np.hstack((A_repeated, B_tiled))

        def cartesian_row(A: Matrix, B: Matrix) -> Matrix:
            A_tiled = np.tile(A, (B.shape[0], 1))
            B_repeated = np.repeat(B, repeats=A.shape[0], axis=0)
            return np.hstack((A_tiled, B_repeated))

        if row_major:
            return functools.reduce(cartesian_row, indices)
        return functools.reduce(cartesian_column, indices)

    @staticmethod
    def points_to_indices(points: Matrix) -> tuple[list[Matrix], list[Matrix]]:
        if points.ndim == 1:
            points = points.reshape(1, -1)
        sites = points.shape[1]
        I_l = [points[:, :k] for k in range(sites)]
        I_g = [points[:, (k + 1) : sites] for k in range(sites)]
        return I_l, I_g


@dataclasses.dataclass
class CrossResults:
    """
    Dataclass containing the results from TCI. Keeps track of values for every iteration (half-sweep).

    Parameters
    ----------
    mps : MPS
        The resulting MPS interpolation of the black-box function.
    errors : Vector
        Vector of error values.
    bonds : Matrix
        Matrix of intermediate bond dimensions.
    times : Vector
        Vector of cumulative computation times.
    evals : Vector
        Vector of cumulative function evaluations.
    """

    mps: MPS
    errors: Vector
    bonds: Matrix
    times: Vector
    evals: Vector


class CrossError:
    """
    Auxiliar base class for TCI used to compute the sampled error between the function and the
    MPS at every iteration using the sampled Lp norm and caching intermediate results for efficiency.
    """

    error_norm: float
    num_samples: int
    error_relative: bool
    rng: np.random.Generator
    mps_indices: Vector | None
    black_box_evals: np.ndarray | None
    norm: float

    def __init__(self, cross_strategy: CrossStrategy):
        self.error_norm = cross_strategy.error_norm
        self.num_samples = cross_strategy.num_samples
        self.error_relative = cross_strategy.error_relative
        self.rng = cross_strategy.rng
        # Cache
        self.mps_indices = None
        self.black_box_evals = None
        self.norm = 1.0

    def lp_distance(self, x: Vector) -> float:
        p = self.error_norm
        if np.isfinite(p):
            dist = ((1 / len(x)) * np.sum(np.abs(x) ** p)) ** (1 / p)
        else:
            dist = np.max(np.abs(x))
        return float(dist)

    def sample_error(self, cross: CrossInterpolation) -> float:
        if self.mps_indices is None:
            # Consider the allowed indices to impose restrictions (e.g. diagonal MPO)
            allowed_indices = getattr(cross.black_box, "allowed_indices", None)
            self.mps_indices = random_mps_indices(
                cross.black_box.physical_dimensions,
                self.num_samples,
                allowed_indices,
                self.rng,
            )
            self.black_box_evals = cross.black_box[self.mps_indices].reshape(-1)
            self.norm = self.lp_distance(self.black_box_evals)
        mps_evals = evaluate_mps(cross.mps, self.mps_indices)
        error = self.lp_distance(mps_evals - self.black_box_evals)
        return error / self.norm if self.error_relative else error


def check_convergence(
    half_sweep: int, trajectories: dict, cross_strategy: CrossStrategy
) -> bool:
    """Checks the convergence of TCI from its trajectories and logs the results for each iteration."""
    maxbond = np.max(trajectories["bonds"][-1])
    maxbond_prev = np.max(trajectories["bonds"][-2]) if half_sweep > 2 else 0
    time = np.sum(trajectories["times"])
    evals = trajectories["evals"][-1]
    with make_logger(2) as logger:
        logger(
            f"Iteration (half-sweep): {half_sweep:3}/{cross_strategy.max_iters}, "
            f"error: {trajectories['errors'][-1]:1.15e}/{cross_strategy.tol:.2e}, "
            f"maxbond: {maxbond:3}/{cross_strategy.max_bond}, "
            f"time: {time:8.6f}/{cross_strategy.max_time}, "
            f"evals: {evals:8}/{cross_strategy.max_evals}."
        )

    if trajectories["errors"][-1] <= cross_strategy.tol:
        logger(f"State converged within tolerance {cross_strategy.tol}")
        return True
    elif maxbond >= cross_strategy.max_bond:
        logger(f"Max. bond reached above the threshold {cross_strategy.max_bond}")
        return True
    elif cross_strategy.max_time is not None and time >= cross_strategy.max_time:
        logger(f"Max. time reached above the threshold {cross_strategy.max_time}")
        return True
    elif cross_strategy.max_evals is not None and evals >= cross_strategy.max_evals:
        logger(f"Max. evals reached above the threshold {cross_strategy.max_evals}")
        return True
    elif maxbond - maxbond_prev <= 0:
        logger(f"Max. bond dimension converged with value {maxbond}")
        return True

    return False


def maxvol_square(
    A: Matrix,
    max_iter: int = 10,
    tol: float = 1.05,
) -> tuple[Matrix, Matrix]:
    """
    Returns the row indices I of a tall matrix A of size (n x r) with n > r that give place
    to a square submatrix of (quasi-)maximum volume (modulus of the submatrix determinant).
    Also, returns a matrix of coefficients B such that A ≈ B A[I, :].

    Parameters
    ----------
    A : np.ndarray
        A tall (n x r) matrix with more rows than columns (n > r).
    maxiter : int, default=100
        Maximum number of iterations allowed.
    tol : float, default=1.1
        Sensibility of the algorithm.

    Returns
    -------
    I : np.ndarray
        An array of r indices that determine a square submatrix of A with (quasi-)maximum volume.
    B : np.ndarray
        A (r x r) submatrix of coefficients such that A ≈ B A[I, :].
    """
    n, r = A.shape
    if n <= r:
        I, B = np.arange(n, dtype=int), np.eye(n)
        return I, B
    P, L, U = scipy.linalg.lu(A)
    I = P[:, :r].argmax(axis=0)
    Q = scipy.linalg.solve_triangular(U, A.T, trans=1)
    B = scipy.linalg.solve_triangular(
        L[:r, :], Q, trans=1, unit_diagonal=True, lower=True
    ).T
    for _ in range(max_iter):
        i, j = np.divmod(abs(B).argmax(), r)
        if abs(B[i, j]) <= tol:
            break
        I[j] = i
        bj = B[:, j]
        bi = B[i, :].copy()
        bi[j] -= 1.0
        B -= np.outer(bj, bi / B[i, j])
    return I, B
