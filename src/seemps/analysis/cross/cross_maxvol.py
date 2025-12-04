import numpy as np
import scipy.linalg
from dataclasses import dataclass
from collections import defaultdict
from time import perf_counter
from typing import Callable, Any

from ...typing import Matrix
from ...tools import make_logger
from .black_box import BlackBox
from .cross import (
    CrossStrategy,
    CrossInterpolation,
    CrossResults,
    CrossError,
    check_convergence,
    maxvol_square,
)


@dataclass
class CrossStrategyMaxvol(CrossStrategy):
    rank_kick: tuple = (0, 1)
    max_iters_maxvol: int = 10
    tol_maxvol_square: float = 1.05
    tol_maxvol_rect: float = 1.05
    """
    Dataclass containing the parameters for the rectangular maxvol-based TCI.
    The common parameters are documented in the base `CrossStrategy` class.
    
    Parameters
    ----------
    rank_kick : tuple, default=(0, 1)
        Minimum and maximum rank increase or 'kick' at each rectangular maxvol decomposition.
    max_iters_maxvol : int, default=10
        Maximum number of iterations for the square maxvol decomposition.
    tol_maxvol_square : float, default=1.05
        Sensibility for the square maxvol decomposition.
    tol_maxvol_rect : float, default=1.05
        Sensibility for the rectangular maxvol decomposition.
    """

    @property
    def algorithm(self) -> Callable:
        return cross_maxvol


def cross_maxvol(
    black_box: BlackBox,
    cross_strategy: CrossStrategyMaxvol = CrossStrategyMaxvol(),
    initial_points: Matrix | None = None,
) -> CrossResults:
    """
    Computes the MPS representation of a black-box function using the tensor cross-approximation (TCI)
    algorithm based on one-site optimizations using the rectangular maxvol decomposition.
    The black-box function can represent several different structures. See `black_box` for usage examples.

    Parameters
    ----------
    black_box : BlackBox
        The black box to approximate as a MPS.
    cross_strategy : CrossStrategyMaxvol = CrossStrategyMaxvol()
        A dataclass containing the parameters of the algorithm.
    initial_points : Optional[Matrix], default=None
        A collection of initial points used to initialize the algorithm.
        If None, the point at origin is used.

    Returns
    -------
    CrossResults
        A dataclass containing the MPS representation of the black-box function,
        among other useful information.
    """
    cross = CrossInterpolation(black_box, initial_points)
    error_calculator = CrossError(cross_strategy)

    converged = False
    trajectories: defaultdict[str, list[Any]] = defaultdict(list)
    for i in range(cross_strategy.max_iters // 2):
        tick = perf_counter()

        # Left-to-right half sweep
        for k in range(cross.sites):
            _update_cross(cross, k, True, cross_strategy)

        # Right-to-left half sweep
        for k in reversed(range(cross.sites)):
            _update_cross(cross, k, False, cross_strategy)

        sweep_time = perf_counter() - tick
        trajectories["errors"].append(error_calculator.sample_error(cross))
        trajectories["bonds"].append(cross.mps.bond_dimensions())
        trajectories["times"].append(sweep_time)
        trajectories["evals"].append(cross.black_box.evals)
        if converged := check_convergence(2 * (i + 1), trajectories, cross_strategy):
            break

    if not converged:
        with make_logger(2) as logger:
            logger("Maximum number of iterations reached")

    return CrossResults(
        mps=cross.mps,
        errors=np.array(trajectories["errors"]),
        bonds=np.array(trajectories["bonds"]),
        times=np.cumsum(trajectories["times"]),
        evals=np.array(trajectories["evals"]),
    )


def _update_cross(
    cross: CrossInterpolation,
    k: int,
    left_to_right: bool,
    cross_strategy: CrossStrategyMaxvol,
) -> None:
    fiber = cross.sample_fiber(k)
    r_l, s, r_g = fiber.shape

    if left_to_right:
        C = fiber.reshape(r_l * s, r_g, order="F")
        Q, _ = scipy.linalg.qr(C, mode="economic", overwrite_a=True, check_finite=False)  # type: ignore
        I, _ = _choose_maxvol(
            Q,  # type: ignore
            cross_strategy.rank_kick,
            cross_strategy.max_iters_maxvol,
            cross_strategy.tol_maxvol_square,
            cross_strategy.tol_maxvol_rect,
        )
        if k < cross.sites - 1:
            cross.I_l[k + 1] = cross.combine_indices(
                cross.I_l[k], cross.I_s[k], row_major=True
            )[I]

    else:
        if k > 0:
            R = fiber.reshape(r_l, s * r_g, order="F")
            Q, _ = scipy.linalg.qr(
                R.T, mode="economic", overwrite_a=True, check_finite=False
            )  # type: ignore
            I, G = _choose_maxvol(
                Q,  # type: ignore
                cross_strategy.rank_kick,
                cross_strategy.max_iters_maxvol,
                cross_strategy.tol_maxvol_square,
                cross_strategy.tol_maxvol_rect,
            )
            cross.mps[k] = (G.T).reshape(-1, s, r_g, order="F")
            cross.I_g[k - 1] = cross.combine_indices(
                cross.I_s[k], cross.I_g[k], row_major=True
            )[I]
        else:
            cross.mps[0] = fiber


def _choose_maxvol(
    A: Matrix,
    rank_kick: tuple,
    max_iter: int,
    tol: float,
    tol_rect: float,
) -> tuple[Matrix, Matrix]:
    n, r = A.shape
    min_kick, max_kick = rank_kick
    max_kick = min(max_kick, n - r)
    min_kick = min(min_kick, max_kick)
    if n <= r:
        I, B = np.arange(n, dtype=int), np.eye(n)
    elif rank_kick == 0:
        I, B = maxvol_square(A, max_iter, tol)
    else:
        I, B = maxvol_rectangular(A, (min_kick, max_kick), max_iter, tol, tol_rect)
    return I, B


def maxvol_rectangular(
    A: Matrix,
    rank_kick: tuple = (0, 1),
    max_iter: int = 10,
    tol: float = 1.05,
    tol_rect: float = 1.05,
) -> tuple[Matrix, Matrix]:
    # TODO: Add a docstring
    n, r = A.shape
    min_rank = r + rank_kick[0]
    max_rank = min(r + rank_kick[1], n)
    if min_rank < r or min_rank > max_rank or max_rank > n:
        raise ValueError("Invalid minimum/maximum number of added rows")
    I0, B = maxvol_square(A, max_iter, tol)
    I = np.hstack([I0, np.zeros(max_rank - r, dtype=I0.dtype)])
    S = np.ones(n, dtype=int)
    S[I0] = 0
    F = S * np.linalg.norm(B) ** 2
    for k in range(r, max_rank):
        i = np.argmax(F)
        if k >= min_rank and F[i] <= tol_rect**2:
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
