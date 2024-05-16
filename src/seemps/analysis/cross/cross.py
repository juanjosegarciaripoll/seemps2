import numpy as np
import dataclasses
import functools

from typing import Optional
from copy import deepcopy
from scipy.linalg import lu, solve_triangular  # type: ignore

from .black_box import BlackBox
from ..sampling import evaluate_mps, random_mps_indices
from ...state import MPS, random_mps
from ...tools import make_logger
from ...typing import VectorLike


@dataclasses.dataclass
class CrossStrategy:
    maxiter: int = 100
    maxbond: int = 1000
    tol_sampling: float = 1e-10
    norm_sampling: float = np.inf
    num_samples: int = 1000
    check_norm_2: bool = False
    tol_norm_2: float = 1e-10
    rng: np.random.Generator = dataclasses.field(
        default_factory=lambda: np.random.default_rng()
    )
    """
    Dataclass containing the base parameters for TCI.

    Parameters
    ----------
    maxiter : int, default=100
        Maximum number of sweeps allowed.
    maxbond : int, default=1000
        Maximum MPS bond dimension allowed.
    tol_sampling : float, default=1e-10
        Tolerance for the sampled error.
    norm_sampling : float, default=np.inf
        Norm used to measure the sampled error.
    num_samples : int, default=1000
        Number of function samples to evaluate the error.
    check_norm_2 : bool, default=True
        Whether to check the change in norm-2 of the MPS after each sweep.
    tol_norm_2 : float, default=1e-10
        Tolerance for the change in norm-2 of the MPS.
    rng : np.random.Generator, default=np.random.default_rng()
        Random number generator used to initialize the algorithm and sample the error.
    """


@dataclasses.dataclass
class CrossResults:
    mps: MPS
    evals: int
    trajectories: Optional[VectorLike] = None


class CrossInterpolation:
    def __init__(
        self,
        black_box: BlackBox,
        initial_point: np.ndarray,
    ):
        self.black_box = black_box
        self.sites = black_box.sites
        self.I_l, self.I_g = self.points_to_indices(initial_point)
        self.I_s = [np.arange(s).reshape(-1, 1) for s in black_box.physical_dimensions]
        # Placeholders
        self.mps = random_mps(black_box.physical_dimensions)
        self.previous_mps: MPS = deepcopy(self.mps)
        self.previous_error: float = np.inf
        self.sample_indices: Optional[np.ndarray] = None
        self.func_samples: Optional[np.ndarray] = None

    def sample_error(
        self,
        num_samples: int,
        norm_error: float,
        allowed_indices: Optional[list[int]] = None,
        rng: np.random.Generator = np.random.default_rng(),
    ) -> float:
        if self.sample_indices is None:
            self.sample_indices = random_mps_indices(
                self.mps,
                num_indices=num_samples,
                allowed_indices=allowed_indices,
                rng=rng,
            )
        if self.func_samples is None:
            self.func_samples = self.black_box[self.sample_indices].reshape(-1)
        mps_samples = evaluate_mps(self.mps, self.sample_indices)
        error = np.linalg.norm(self.func_samples - mps_samples, ord=norm_error)  # type: ignore
        prefactor = np.prod(self.func_samples.shape) ** (1 / norm_error)
        return error / prefactor  # type: ignore

    def norm_2_change(self) -> float:
        change_norm = (self.mps - self.previous_mps).norm() / self.previous_mps.norm()
        self.previous_mps = deepcopy(self.mps)
        return change_norm

    @staticmethod
    def points_to_indices(points: np.ndarray) -> tuple[list, list]:
        if points.ndim == 1:
            points = points.reshape(1, -1)
        sites = points.shape[1]
        I_l = [points[:, :k] for k in range(sites)]
        I_g = [points[:, (k + 1) : sites] for k in range(sites)]
        return I_l, I_g

    @staticmethod
    def combine_indices(*indices: np.ndarray) -> np.ndarray:
        """
        Computes the Cartesian product of a set of multi-indices arrays and arranges the
        result as concatenated indices in C order.

        Parameters
        ----------
        indices : *np.ndarray
            A variable number of arrays where each array is treated as a set of multi-indices.

        Example
        -------
        >>> combine_indices(np.array([[1, 2, 3], [4, 5, 6]]), np.array([[0], [1]]))
        array([[1, 2, 3, 0],
               [1, 2, 3, 1],
               [4, 5, 6, 0],
               [4, 5, 6, 1]])
        """

        # TODO: Compute a collection of rows of the cartesian product directly without first
        # computing the whole cartesian product.
        def cartesian(A: np.ndarray, B: np.ndarray) -> np.ndarray:
            A_repeated = np.repeat(A, repeats=B.shape[0], axis=0)
            B_tiled = np.tile(B, (A.shape[0], 1))
            return np.hstack((A_repeated, B_tiled))

        return functools.reduce(cartesian, indices)


def maxvol_square(
    A: np.ndarray, maxiter: int = 100, tol: float = 1.1
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns the row indices I of a tall matrix A of size (n x r) with n > r that give place
    to a square submatrix of (quasi-)maximum volume (modulus of the submatrix determinant).
    Also, returns a matrix of coefficients B such that A ≈ B A[I, :].

    Parameters
    ----------
    A : np.ndarray
        A tall (n x r) matrix with more rows than columns (n > r).
    maxiter : int, default = 100
        Maximum number of iterations allowed.
    tol : float, default = 1.05
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
    P, L, U = lu(A, check_finite=False)  # type: ignore
    I = P[:, :r].argmax(axis=0)
    Q = solve_triangular(U, A.T, trans=1, check_finite=False)
    B = solve_triangular(
        L[:r, :], Q, trans=1, check_finite=False, unit_diagonal=True, lower=True
    ).T
    for _ in range(maxiter):
        i, j = np.divmod(abs(B).argmax(), r)
        if abs(B[i, j]) <= tol:
            break
        I[j] = i
        bj = B[:, j]
        bi = B[i, :].copy()
        bi[j] -= 1.0
        B -= np.outer(bj, bi / B[i, j])
    return I, B


def _check_convergence(
    cross: CrossInterpolation,
    sweep: int,
    cross_strategy: CrossStrategy,
) -> tuple[bool, str]:
    allowed_sampling_indices = getattr(
        cross.black_box, "allowed_sampling_indices", None
    )
    error = cross.sample_error(
        cross_strategy.num_samples,
        cross_strategy.norm_sampling,
        allowed_indices=allowed_sampling_indices,
        rng=cross_strategy.rng,
    )
    maxbond = max(cross.mps.bond_dimensions())
    logger = make_logger()
    logger(
        f"Cross sweep {1+sweep:3d} with error({cross_strategy.num_samples} samples "
        f"in norm-{cross_strategy.norm_sampling})={error}, maxbond={maxbond}, evals(cumulative)={cross.black_box.evals}"
    )
    converged = False
    message = f"Maximum number of sweeps {cross_strategy.maxiter} reached"
    if cross_strategy.check_norm_2:
        change_norm = cross.norm_2_change()
        logger(f"Norm-2 change {change_norm}")
        if change_norm <= cross_strategy.tol_norm_2:
            converged = True
            message = f"Stationary state reached with norm-2 change {change_norm}"
    if error < cross_strategy.tol_sampling:
        converged = True
        message = f"State converged within tolerance {cross_strategy.tol_sampling}"
    elif maxbond > cross_strategy.maxbond:
        converged = True
        message = f"Maxbond reached above the threshold {cross_strategy.maxbond}"
    return converged, message
