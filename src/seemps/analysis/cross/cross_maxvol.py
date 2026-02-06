from __future__ import annotations
import numpy as np
import scipy.linalg
from dataclasses import dataclass
from ...typing import Matrix
from .black_box import BlackBox
from .cross import (
    CrossStrategy,
    CrossInterpolation,
    CrossResults,
    maxvol_square,
    cross_interpolation,
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

    def make_interpolator(
        self, black_box: BlackBox, initial_points: Matrix | None = None
    ) -> CrossInterpolation:
        return CrossInterpolationMaxvol(self, black_box, initial_points)


class CrossInterpolationMaxvol(CrossInterpolation):
    strategy: CrossStrategyMaxvol

    def __init__(
        self,
        strategy: CrossStrategyMaxvol,
        black_box: BlackBox,
        initial_points: Matrix | None = None,
    ):
        super().__init__(
            black_box,
            initial_points,
            two_sweeps_required=True,
            two_site_algorithm=False,
        )
        self.strategy = strategy

    def update(self, k: int, left_to_right: bool) -> None:
        cross_strategy = self.strategy
        fiber = self.sample_fiber(k)
        r_l, s, r_g = fiber.shape

        if left_to_right:
            C = fiber.reshape(r_l * s, r_g, order="F")
            Q, _ = scipy.linalg.qr(
                C, mode="economic", overwrite_a=True, check_finite=False
            )  # type: ignore
            I, _ = _choose_maxvol(
                Q,  # type: ignore
                cross_strategy.rank_kick,
                cross_strategy.max_iters_maxvol,
                cross_strategy.tol_maxvol_square,
                cross_strategy.tol_maxvol_rect,
            )
            if k < self.sites - 1:
                self.I_l[k + 1] = self.combine_indices(
                    self.I_l[k], self.I_s[k], row_major=True
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
                self.mps[k] = (G.T).reshape(-1, s, r_g, order="F")
                self.I_g[k - 1] = self.combine_indices(
                    self.I_s[k], self.I_g[k], row_major=True
                )[I]
            else:
                self.mps[0] = fiber


def cross_maxvol(
    black_box: BlackBox,
    initial_points: Matrix | None = None,
    cross_strategy: CrossStrategyMaxvol = CrossStrategyMaxvol(),
) -> CrossResults:
    """
    Computes the MPS representation of a black-box function using the tensor cross-approximation (TCI)
    algorithm based on one-site optimizations using the rectangular maxvol decomposition.
    The black-box function can represent several different structures. See `black_box` for usage examples.

    Parameters
    ----------
    black_box : BlackBox
        The black box to approximate as a MPS.
    initial_points : Optional[Matrix], default=None
        A collection of initial points used to initialize the algorithm.
        If None, the point at origin is used.
    cross_strategy : CrossStrategyMaxvol = CrossStrategyMaxvol()
        A dataclass containing the parameters of the algorithm.

    Returns
    -------
    CrossResults
        A dataclass containing the MPS representation of the black-box function,
        among other useful information.
    """
    return cross_interpolation(cross_strategy, black_box, initial_points)


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
