import numpy as np
import dataclasses
import functools

from .cross import (
    CrossInterpolation,
    CrossResults,
    CrossStrategy,
    BlackBox,
    maxvol_square,
    _check_convergence,
)
from ...state._contractions import _contract_last_and_first
from ...tools import make_logger


@dataclasses.dataclass
class CrossStrategyMaxvol(CrossStrategy):
    maxvol_tol: float = 1.1
    maxvol_maxiter: int = 100
    maxvol_rect_tol: float = 1.1
    maxvol_rect_rank_change: tuple = (1, np.inf)
    fortran_order: bool = True
    """
    Dataclass containing the parameters for the maxvol-based TCI.
    The common parameters are documented in the base `CrossStrategy` class.

    Parameters
    ----------
    maxvol_tol : float, default = 1.1
        Sensibility for the square maxvol decomposition.
    maxvol_maxiter : int, default = 100
        Maximum number of iterations for the square maxvol decomposition.
    maxvol_rect_tol : float, default = 1.1
        Sensibility for the rectangular maxvol decomposition.
    maxvol_rect_rank_change : tuple, default = (1, np.inf)
        Minimum and maximum increase allowed for the bond dimension at each half sweep.
    fortran_order: bool, default = True
        Whether to use the Fortran order in the computation of the maxvol indices.
        For some reason, the Fortran order converges better for some functions.
    """


class CrossInterpolationMaxvol(CrossInterpolation):
    def __init__(self, black_box: BlackBox, initial_point: np.ndarray):
        super().__init__(black_box, initial_point)

    def sample_fiber(self, k: int) -> np.ndarray:
        i_l, i_s, i_g = self.I_l[k], self.I_s[k], self.I_g[k]
        mps_indices = self.combine_indices(i_l, i_s, i_g)
        return self.black_box[mps_indices].reshape((len(i_l), len(i_s), len(i_g)))

    @staticmethod
    def combine_indices_fortran(*indices: np.ndarray) -> np.ndarray:
        """
        Computes the Cartesian product of a set of multi-indices arrays and arranges the
        result as concatenated indices in Fortran order (column-major).

        Parameters
        ----------
        indices : *np.ndarray
            A variable number of arrays where each array is treated as a set of multi-indices.
        fortran_order : bool, default=False
            If True, the output is arranged in Fortran order where the first index changes fastest.
            If False, the output is arranged in C order where the last index changes fastest.

        Example
        -------
        >>> combine_indices(np.array([[1, 2, 3], [4, 5, 6]]), np.array([[0], [1]]), fortran_order=True)
        array([[1, 2, 3, 0],
               [4, 5, 6, 0],
               [1, 2, 3, 1],
               [4, 5, 6, 1]])
        """

        def cartesian_fortran(A: np.ndarray, B: np.ndarray) -> np.ndarray:
            A_tiled = np.tile(A, (B.shape[0], 1))
            B_repeated = np.repeat(B, repeats=A.shape[0], axis=0)
            return np.hstack((A_tiled, B_repeated))

        return functools.reduce(cartesian_fortran, indices)


def cross_maxvol(
    black_box: BlackBox,
    cross_strategy: CrossStrategyMaxvol = CrossStrategyMaxvol(),
) -> CrossResults:
    """
    Computes the MPS representation of a black-box function using the tensor cross-approximation (TCI)
    algorithm based on one-site optimizations using the rectangular maxvol decomposition.

    Parameters
    ----------
    black_box : BlackBox
        The black box to approximate as a MPS.
    cross_strategy : CrossStrategy, default=CrossStrategy()
        A dataclass containing the parameters of the algorithm.

    Returns
    -------
    mps : MPS
        The MPS representation of the black-box function.
    """
    initial_point = cross_strategy.rng.integers(
        low=0, high=black_box.base, size=black_box.sites
    )
    cross = CrossInterpolationMaxvol(black_box, initial_point)
    converged = False
    with make_logger(2) as logger:
        for i in range(cross_strategy.maxiter):
            # Forward sweep
            for k in range(cross.sites):
                _update_maxvol(cross, k, True, cross_strategy)
            # Backward sweep
            for k in reversed(range(cross.sites)):
                _update_maxvol(cross, k, False, cross_strategy)
            if converged := _check_convergence(cross, i, cross_strategy, logger):
                break
        if not converged:
            logger("Maximum number of iterations reached")
    return CrossResults(mps=cross.mps, evals=black_box.evals)


def _update_maxvol(
    cross: CrossInterpolationMaxvol,
    k: int,
    forward: bool,
    cross_strategy: CrossStrategyMaxvol,
) -> None:
    if cross_strategy.fortran_order is True:
        combine_indices = cross.combine_indices_fortran
        order = "F"
    else:
        combine_indices = cross.combine_indices
        order = "C"
    fiber = cross.sample_fiber(k)
    r_l, s, r_g = fiber.shape
    if forward:
        C = fiber.reshape(r_l * s, r_g, order=order)  # type: ignore
        Q, _ = np.linalg.qr(C)
        I, _ = choose_maxvol(
            Q,
            cross_strategy.maxvol_maxiter,
            cross_strategy.maxvol_tol,
            cross_strategy.maxvol_rect_tol,
            cross_strategy.maxvol_rect_rank_change,
        )
        if k < cross.sites - 1:
            cross.I_l[k + 1] = combine_indices(cross.I_l[k], cross.I_s[k])[I]
    else:
        R = fiber.reshape(r_l, s * r_g, order=order)  # type: ignore
        Q, T = np.linalg.qr(R.T)
        I, G = choose_maxvol(
            Q,
            cross_strategy.maxvol_maxiter,
            cross_strategy.maxvol_tol,
            cross_strategy.maxvol_rect_tol,
            cross_strategy.maxvol_rect_rank_change,
        )
        cross.mps[k] = (G.T).reshape(-1, s, r_g, order=order)  # type: ignore
        if k > 0:
            cross.I_g[k - 1] = combine_indices(cross.I_s[k], cross.I_g[k])[I]
        elif k == 0:
            cross.mps[0] = _contract_last_and_first((Q[I] @ T).T, cross.mps[0])


def choose_maxvol(
    A: np.ndarray,
    maxiter: int = 100,
    tol: float = 1.1,
    tol_rect: float = 1.05,
    rank_change: tuple = (1, 1),
) -> tuple[np.ndarray, np.ndarray]:
    n, r = A.shape
    min_rank_change, max_rank_change = rank_change
    max_rank_change = min(max_rank_change, n - r)
    min_rank_change = min(min_rank_change, max_rank_change)
    if n <= r:
        I, B = np.arange(n, dtype=int), np.eye(n)
    elif max_rank_change == 0:
        I, B = maxvol_square(A, maxiter, tol)
    else:
        I, B = maxvol_rectangular(
            A, maxiter, tol, min_rank_change, max_rank_change, tol_rect
        )
    return I, B


def maxvol_rectangular(
    A: np.ndarray,
    maxiter: int = 100,
    tol: float = 1.1,
    min_rank_change: int = 1,
    max_rank_change: int = 1,
    tol_rect: float = 1.05,
):
    n, r = A.shape
    r_min = r + min_rank_change
    r_max = r + max_rank_change if max_rank_change is not None else n
    r_max = min(r_max, n)
    if r_min < r or r_min > r_max or r_max > n:
        raise ValueError("Invalid minimum/maximum number of added rows")
    I0, B = maxvol_square(A, maxiter, tol)
    I = np.hstack([I0, np.zeros(r_max - r, dtype=I0.dtype)])
    S = np.ones(n, dtype=int)
    S[I0] = 0
    F = S * np.linalg.norm(B, axis=1) ** 2
    for k in range(r, r_max):
        i = np.argmax(F)
        if k >= r_min and F[i] <= tol_rect**2:
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
