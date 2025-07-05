import numpy as np
import scipy.linalg  # type: ignore
import dataclasses
import functools
from copy import deepcopy
from typing import TypeAlias
from .black_box import BlackBox
from ..sampling import evaluate_mps, random_mps_indices
from ...state import MPS, random_mps
from ...tools import Logger
from ...typing import VectorLike, Natural


@dataclasses.dataclass
class CrossStrategy:
    maxiter: Natural = 100
    maxbond: int = 1000
    tol_sampling: float = 1e-10
    norm_sampling: float = np.inf
    num_samples: Natural = 1000
    tol_norm_2: float | None = None
    rng: np.random.Generator = dataclasses.field(
        default_factory=lambda: np.random.default_rng()
    )
    """
    Dataclass containing the base parameters for TCI.

    Parameters
    ----------
    maxiter : int (> 0), default=100
        Maximum number of sweeps allowed.
    maxbond : int, default=1000
        Maximum MPS bond dimension allowed.
    tol_sampling : float, default=1e-12
        Tolerance for the sampled error.
    norm_sampling : float, default=np.inf
        Norm used to measure the sampled error.
    num_samples : int, default=1000
        Number of function samples to evaluate the error.
    tol_norm_2 : float, optional
        Tolerance for the increment in norm-2 of the MPS after each sweep.
        If None, this increment is not measured.
    rng : np.random.Generator, default=np.random.default_rng()
        Random number generator used to initialize the algorithm and sample the error.
    """

    def __post_init__(self) -> None:
        assert self.maxiter > 0
        assert self.num_samples > 0


IndexMatrix: TypeAlias = np.ndarray[tuple[int, int], np.dtype[np.integer]]
IndexVector: TypeAlias = np.ndarray[tuple[int], np.dtype[np.integer]]
IndexSlice: TypeAlias = np.intp | IndexVector | slice


@dataclasses.dataclass
class CrossResults:
    """
    Dataclass containing the results from TCI.

    Parameters
    ----------
    mps : MPS
        The resulting MPS interpolation of the black-box function.
    evals : int
        The number of function evaluations required for the interpolation.
    points : np.ndarray
        The indices of the discretization points whose multivariate crosses yield
        the interpolation.
    callback_output : VectorLike, optional
        An array collecting the results of the callback function, called at each iteration.
    trajectories : VectorLike, optional
        A collection of arrays containing information of the interpolation for each iteration.
    """

    mps: MPS
    evals: int
    points: np.ndarray
    callback_output: VectorLike | None = None
    trajectories: VectorLike | None = None


class CrossInterpolation:
    """Auxiliar base class for TCI used to keep track of the required
    interpolation information."""

    black_box: BlackBox
    sites: int
    I_l: list  # TODO: More precise annotation
    I_g: list
    I_s: list[np.ndarray]
    mps: MPS
    previous_mps: MPS
    previous_error: float
    mps_indices: np.ndarray | None
    func_samples: np.ndarray | None

    def __init__(
        self,
        black_box: BlackBox,
        initial_points: np.ndarray,
    ):
        self.black_box = black_box
        self.sites = black_box.sites
        self.I_l, self.I_g = self.points_to_indices(initial_points)
        self.I_s = [np.arange(s).reshape(-1, 1) for s in black_box.physical_dimensions]
        # Placeholders
        self.mps = random_mps(black_box.physical_dimensions)
        self.previous_mps = deepcopy(self.mps)
        self.previous_error = np.inf
        self.mps_indices = None
        self.func_samples = None

    def sample_fiber(self, k: int) -> np.ndarray:
        i_l, i_s, i_g = self.I_l[k], self.I_s[k], self.I_g[k]
        mps_indices = self.combine_indices(i_l, i_s, i_g)
        return self.black_box[mps_indices].reshape((len(i_l), len(i_s), len(i_g)))

    def sample_error(
        self,
        num_samples: int,
        norm_error: float,
        allowed_indices: list[int] | None = None,
        rng: np.random.Generator = np.random.default_rng(),
    ) -> float:
        if self.mps_indices is None:
            self.mps_indices = random_mps_indices(
                self.mps.physical_dimensions(),
                num_indices=num_samples,
                allowed_indices=allowed_indices,
                rng=rng,
            )
        if self.func_samples is None:
            self.func_samples = self.black_box[self.mps_indices].reshape(-1)
        mps_samples = evaluate_mps(self.mps, self.mps_indices)
        error = np.linalg.norm(self.func_samples - mps_samples, ord=norm_error)
        prefactor = np.prod(self.func_samples.shape) ** (1 / norm_error)
        return float(error / prefactor)

    def norm_2_increment(self) -> float:
        norm_increment = (
            (self.mps - self.previous_mps).norm() / self.previous_mps.norm()
        ) ** 2
        self.previous_mps = deepcopy(self.mps)
        return norm_increment

    def indices_to_points(self, forward: bool) -> np.ndarray:
        """
        Computes the MPS 'points' that result in the best TCI approximation.
        This is done performing a sweep with the square maxvol decomposition.
        """
        if forward:
            for k in range(self.sites):
                fiber = self.sample_fiber(k)
                r_l, s, r_g = fiber.shape
                C = fiber.reshape(r_l * s, r_g)
                Q, _ = np.linalg.qr(C)
                I, _ = maxvol_square(Q)
                if k < self.sites - 1:
                    self.I_l[k + 1] = self.combine_indices(self.I_l[k], self.I_s[k])[I]
                else:
                    indices = self.I_l[1:] + [
                        self.combine_indices(self.I_l[k], self.I_s[k])[I]
                    ]
        else:
            for k in reversed(range(self.sites)):
                fiber = self.sample_fiber(k)
                r_l, s, r_g = fiber.shape
                R = fiber.reshape(r_l, s * r_g)
                Q, _ = np.linalg.qr(R.T)
                I, _ = maxvol_square(Q)
                if k > 0:
                    self.I_g[k - 1] = self.combine_indices(self.I_s[k], self.I_g[k])[I]
                else:
                    # ruff: noqa: F841
                    indices = [
                        self.combine_indices(self.I_s[0], self.I_g[0])[I]
                    ] + self.I_g[:-1]
        # TODO: Get points from indices
        return np.array([])

    @staticmethod
    def points_to_indices(points: np.ndarray) -> tuple[list, list]:
        if points.ndim == 1:
            points = points.reshape(1, -1)
        sites = points.shape[1]
        I_l = [points[:, :k] for k in range(sites)]
        I_g = [points[:, (k + 1) : sites] for k in range(sites)]
        return I_l, I_g

    @staticmethod
    def combine_indices(*indices: IndexMatrix) -> IndexMatrix:
        """
        Computes the Cartesian product of a set of multi-indices arrays and arranges the
        result as concatenated indices in C order (column-major).

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

        # TODO: Avoid computing the whole cartesian product.
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
    P, L, U = scipy.linalg.lu(A, check_finite=False)  # type: ignore
    I = P[:, :r].argmax(axis=0)
    Q = scipy.linalg.solve_triangular(U, A.T, trans=1, check_finite=False)
    B = scipy.linalg.solve_triangular(
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


# WARNING: If this function is to be imported, do not use "_" in front.
def _check_convergence(
    cross: CrossInterpolation,
    sweep: int,
    cross_strategy: CrossStrategy,
    logger: Logger,
) -> bool:
    error = cross.sample_error(
        cross_strategy.num_samples,
        cross_strategy.norm_sampling,
        allowed_indices=getattr(cross.black_box, "allowed_indices", None),
        rng=cross_strategy.rng,
    )
    maxbond = cross.mps.max_bond_dimension()
    evals = cross.black_box.evals - cross_strategy.num_samples  # subtract error evals
    if logger:
        logger(
            f"Cross sweep {1 + sweep:3d} with error({cross_strategy.num_samples} samples "
            + f"in norm-{cross_strategy.norm_sampling})={error}, maxbond={maxbond}, evals(cumulative)={evals}"
        )
    if cross_strategy.tol_norm_2 is not None:
        norm_increment = cross.norm_2_increment()
        logger(f"Norm-2 increment {norm_increment}")
        if norm_increment <= cross_strategy.tol_norm_2:
            logger(f"Stationary state reached with norm-2 increment {norm_increment}")
            return True
    if error < cross_strategy.tol_sampling:
        logger(f"State converged within tolerance {cross_strategy.tol_sampling}")
        return True
    elif maxbond > cross_strategy.maxbond:
        logger(f"Maxbond reached above the threshold {cross_strategy.maxbond}")
        return True
    return False
