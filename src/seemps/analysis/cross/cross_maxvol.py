import numpy as np
import scipy.linalg  # type: ignore
import dataclasses
import functools
from typing import Optional, Callable

from .cross import (
    CrossInterpolation,
    CrossResults,
    CrossStrategy,
    BlackBox,
    maxvol_square,
    _check_convergence,
)
from ..sampling import random_mps_indices
from ...tools import make_logger

# TODO: Implement local error evaluation


@dataclasses.dataclass
class CrossStrategyMaxvol(CrossStrategy):
    rank_kick: tuple = (0, 1)
    maxiter_maxvol_square: int = 10
    tol_maxvol_square: float = 1.05
    tol_maxvol_rect: float = 1.05
    fortran_order: bool = True
    """
    Dataclass containing the parameters for the rectangular maxvol-based TCI.
    The common parameters are documented in the base `CrossStrategy` class.

    Parameters
    ----------
    rank_kick : tuple, default=(0, 1)
        Minimum and maximum rank increase or 'kick' at each rectangular maxvol decomposition.
    maxiter_maxvol_square : int, default=10
        Maximum number of iterations for the square maxvol decomposition.
    tol_maxvol_square : float, default=1.05
        Sensibility for the square maxvol decomposition.
    tol_maxvol_rect : float, default=1.05
        Sensibility for the rectangular maxvol decomposition.
    fortran_order: bool, default=True
        Whether to use the Fortran order in the computation of the maxvol indices.
    """


def cross_maxvol(
    black_box: BlackBox,
    cross_strategy: CrossStrategyMaxvol = CrossStrategyMaxvol(),
    initial_points: Optional[np.ndarray] = None,
    callback: Optional[Callable] = None,
) -> CrossResults:
    """
    Computes the MPS representation of a black-box function using the tensor cross-approximation (TCI)
    algorithm based on one-site optimizations using the rectangular maxvol decomposition.
    The black-box function can represent several different structures. See `black_box` for usage examples.

    Parameters
    ----------
    black_box : BlackBox
        The black box to approximate as a MPS.
    cross_strategy : CrossStrategy, default=CrossStrategy()
        A dataclass containing the parameters of the algorithm.
    initial_points : np.ndarray, optional
        A collection of initial points used to initialize the algorithm.
        If None, an initial random point is used.
    callback : Callable, optional
        A callable called on the MPS after each iteration.
        The output of the callback is included in a list 'callback_output' in CrossResults.

    Returns
    -------
    CrossResults
        A dataclass containing the MPS representation of the black-box function,
        among other useful information.
    """
    if initial_points is None:
        initial_points = random_mps_indices(
            black_box.physical_dimensions,
            num_indices=1,
            allowed_indices=getattr(black_box, "allowed_indices", None),
            rng=cross_strategy.rng,
        )

    cross = CrossInterpolationMaxvol(black_box, initial_points)
    converged = False
    callback_output = []
    with make_logger(2) as logger:
        for i in range(cross_strategy.maxiter):
            # Forward sweep
            for k in range(cross.sites):
                _update_maxvol(cross, k, True, cross_strategy)
            # Backward sweep
            for k in reversed(range(cross.sites)):
                _update_maxvol(cross, k, False, cross_strategy)
            if callback:
                callback_output.append(callback(cross.mps, logger=logger))
            if converged := _check_convergence(cross, i, cross_strategy, logger):
                break
        if not converged:
            logger("Maximum number of iterations reached")
    points = cross.indices_to_points(False)
    return CrossResults(
        mps=cross.mps,
        points=points,
        evals=black_box.evals,
        callback_output=callback_output,
    )


class CrossInterpolationMaxvol(CrossInterpolation):
    def __init__(self, black_box: BlackBox, initial_point: np.ndarray):
        super().__init__(black_box, initial_point)

    @staticmethod
    def combine_indices_fortran(*indices: np.ndarray) -> np.ndarray:
        """
        Computes the Cartesian product of a set of multi-indices arrays and arranges the
        result as concatenated indices in Fortran order (row-major).

        Parameters
        ----------
        indices : *np.ndarray
            A variable number of arrays where each array is treated as a set of multi-indices.

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
        Q, _ = scipy.linalg.qr(C, mode="economic", overwrite_a=True, check_finite=False)  # type: ignore
        I, _ = choose_maxvol(
            Q,  # type: ignore
            cross_strategy.rank_kick,
            cross_strategy.maxiter_maxvol_square,
            cross_strategy.tol_maxvol_square,
            cross_strategy.tol_maxvol_rect,
        )
        if k < cross.sites - 1:
            cross.I_l[k + 1] = combine_indices(cross.I_l[k], cross.I_s[k])[I]
    else:
        if k > 0:
            R = fiber.reshape(r_l, s * r_g, order=order)  # type: ignore
            Q, _ = scipy.linalg.qr(  # type: ignore
                R.T, mode="economic", overwrite_a=True, check_finite=False
            )
            I, G = choose_maxvol(
                Q,  # type: ignore
                cross_strategy.rank_kick,
                cross_strategy.maxiter_maxvol_square,
                cross_strategy.tol_maxvol_square,
                cross_strategy.tol_maxvol_rect,
            )
            cross.mps[k] = (G.T).reshape(-1, s, r_g, order=order)  # type: ignore
            cross.I_g[k - 1] = combine_indices(cross.I_s[k], cross.I_g[k])[I]
        else:
            cross.mps[0] = fiber


def choose_maxvol(
    A: np.ndarray,
    rank_kick: tuple = (0, np.inf),
    maxiter: int = 10,
    tol: float = 1.1,
    tol_rect: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    n, r = A.shape
    max_rank_kick = min(rank_kick[1], n - r)
    min_rank_kick = min(rank_kick[0], max_rank_kick)
    if n < r:
        return np.arange(n, dtype=int), np.eye(n)
    elif rank_kick == 0:
        return maxvol_square(A, maxiter, tol)
    else:
        return maxvol_rectangular(
            A, min_rank_kick, max_rank_kick, maxiter, tol, tol_rect
        )


def maxvol_rectangular(
    A: np.ndarray,
    min_rank_kick: int = 0,
    max_rank_kick: float = np.inf,
    maxiter: int = 10,
    tol: float = 1.1,
    tol_rect: float = 1.05,
):
    n, r = A.shape
    r_min = r + min_rank_kick
    r_max = min(r + max_rank_kick, n)
    if r_min < r or r_min > r_max or r_max > n:
        raise ValueError("Invalid minimum/maximum number of added rows")
    I0, B = maxvol_square(A, maxiter, tol)
    I = np.hstack([I0, np.zeros(r_max - r, dtype=I0.dtype)])  # type: ignore
    S = np.ones(n, dtype=int)
    S[I0] = 0
    F = S * np.linalg.norm(B) ** 2
    for k in range(r, int(r_max)):
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
