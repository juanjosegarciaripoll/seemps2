from __future__ import annotations

import numpy as np
from scipy.linalg import solve_triangular
from dataclasses import dataclass
from typing import Literal

from ...typing import Matrix, Vector
from .black_box import BlackBox
from .kernels import combine_indices, prrlu
from .cross import CrossInterpolation, CrossResults, CrossStrategy, cross_interpolation


@dataclass
class CrossStrategyPRRLU(CrossStrategy):
    """
    Dataclass containing the parameters for PRRLU TCI.
    The common parameters are documented in the base `CrossStrategy` class.

    Parameters
    ----------
    tol_pivot : float, default=1e-12
        Smallest pivot accepted by PRRLU, relative to the largest sampled entry.
    pivot_search : {"auto", "full", "rook"}, default="auto"
        Pivot search on each superblock:
        - "full" samples and factors the whole superblock.
        - "rook" alternately factors sampled columns and rows, starting from the current pivots.
        - "auto" uses full search on superblocks with at most `full_pivot_threshold` entries, rook otherwise.
    maxiter_rook : int, default=3
        Maximum number of alternating factorizations in a rook search.
    full_pivot_threshold : int, default=4096
        Largest superblock size for which "auto" uses full search.
    """

    tol_pivot: float = 1e-12
    pivot_search: Literal["auto", "full", "rook"] = "auto"
    maxiter_rook: int = 3
    full_pivot_threshold: int = 4096

    def make_interpolator(
        self, black_box: BlackBox, initial_points: Matrix | None = None
    ) -> CrossInterpolation:
        return CrossInterpolationPRRLU(self, black_box, initial_points)


class CrossInterpolationPRRLU(CrossInterpolation):
    strategy: CrossStrategyPRRLU

    def __init__(
        self,
        strategy: CrossStrategyPRRLU,
        black_box: BlackBox,
        initial_points: Matrix | None = None,
    ):
        super().__init__(black_box, initial_points, cache_evals=strategy.cache_evals)
        self.strategy = strategy
        if self.sites == 1:
            self.mps[0] = self.sample_fiber(0)

    def update(self, k: int, left_to_right: bool) -> None:
        r_l, s_1 = len(self.I_l[k]), len(self.I_s[k])
        s_2, r_g = len(self.I_s[k + 1]), len(self.I_g[k + 1])

        i_ls = combine_indices(self.I_l[k], self.I_s[k])
        i_sg = combine_indices(self.I_s[k + 1], self.I_g[k + 1])
        I, J, L, U, block = self.search_pivots(k, i_ls, i_sg, left_to_right)

        # Skeleton: A ≈ C A[I, J]⁻¹ R, where C = A[:, J] and R = A[I, :]
        # The factors are finite: PRRLU rejects nonfinite superblocks
        if left_to_right:
            C = solve_triangular(
                L[I], L.T, trans="T", lower=True, unit_diagonal=True, check_finite=False
            ).T
            R = block
        else:
            C = block
            R = (
                solve_triangular(U[:, J], U, check_finite=False)
                if U.any()
                else np.ones_like(U)
            )

        # Set MPS cores
        r = len(I)
        self.mps[k] = C.reshape(r_l, s_1, r)
        self.mps[k + 1] = R.reshape(r, s_2, r_g)
        self.I_l[k + 1], self.I_g[k] = i_ls[I], i_sg[J]

    def search_pivots(
        self, k: int, rows: Matrix, columns: Matrix, left_to_right: bool
    ) -> tuple[Vector, Vector, Matrix, Matrix, Matrix]:
        """PRRLU pivots and factors of the superblock with the given rows and
        columns, and the block that becomes a core: A[I, :] in a left-to-right
        sweep and A[:, J] otherwise."""
        # TODO: Refactor
        strategy = self.strategy
        max_rank = strategy.range_max_bonds[1]

        # The rook search starts from the current pivots, plus as many new ones
        old, candidates = (
            (self.I_g[k], columns) if left_to_right else (self.I_l[k + 1], rows)
        )
        position = {p: n for n, p in enumerate(map(tuple, candidates.tolist()))}
        P = np.array(
            [position[p] for p in map(tuple, old.tolist()) if p in position], dtype=int
        )
        dimensions = self.black_box.physical_dimensions[k : k + 2]
        full = strategy.pivot_search == "full" or (
            strategy.pivot_search == "auto"
            and (
                (
                    max(dimensions) <= 16
                    and len(rows) * len(columns) <= strategy.full_pivot_threshold
                )
                or (
                    all(d == 2 for d in dimensions)
                    and len(P) + max(1, len(old)) >= len(candidates)
                )
            )
        )
        if full:
            A = self.sample(rows, columns).reshape(len(rows), len(columns))
            I, J, L, U = prrlu(A, strategy.tol_pivot * np.max(np.abs(A)), max_rank)
            return I, J, L, U, A[I, :] if left_to_right else A[:, J]

        # Alternately factor the columns A[:, P] and the rows A[P, :], taking P
        # from the pivots of the last factorization, until they repeat. Finish
        # with columns (rows) in a left-to-right (right-to-left) sweep, so that
        # L (U) spans the whole superblock.
        available = np.setdiff1d(np.arange(len(candidates)), P)
        new = min(max(1, len(old)), len(available))
        P = np.concatenate((P, strategy.rng.choice(available, new, replace=False)))
        sampled: dict[bool, dict[int, Vector]] = {True: {}, False: {}}
        by_column, finished, scale, last = left_to_right, False, 0.0, None
        iteration = 0
        while True:
            iteration += 1
            A = self.sample_lines(rows, columns, P, sampled[by_column], by_column)
            scale = max(scale, np.max(np.abs(A)))
            I, J, L, U = prrlu(A, strategy.tol_pivot * scale, max_rank)
            I, J = (I, P[J]) if by_column else (P[I], J)
            repeated = (
                last is not None
                and np.array_equal(I, last[0])
                and np.array_equal(J, last[1])
            )
            finished = finished or repeated or iteration >= strategy.maxiter_rook
            if finished and by_column == left_to_right:
                break
            last = I, J
            P = I if by_column else J
            by_column = not by_column
        lines = I if left_to_right else J
        block = self.sample_lines(
            rows, columns, lines, sampled[not left_to_right], not left_to_right
        )
        return I, J, L, U, block

    def sample_lines(
        self,
        rows: Matrix,
        columns: Matrix,
        lines: Vector,
        sampled: dict[int, Vector],
        by_column: bool,
    ) -> Matrix:
        """Sample the superblock columns (or rows) `lines`, reusing `sampled` ones."""
        # TODO: Refactor
        missing = [n for n in lines if n not in sampled]
        if missing:
            if by_column:
                values = self.sample(rows, columns[missing]).reshape(len(rows), -1).T
            else:
                values = self.sample(rows[missing], columns).reshape(len(missing), -1)
            sampled.update(zip(missing, values))
        values = np.array([sampled[n] for n in lines])
        return values.T if by_column else values


def cross_prrlu(
    black_box: BlackBox,
    initial_points: Matrix | None = None,
    cross_strategy: CrossStrategyPRRLU | None = None,
) -> CrossResults:
    """
    Computes the MPS representation of a black-box function using the tensor cross-interpolation (TCI)
    algorithm based on rank-adaptive partial rank-revealing LU (PRRLU) decompositions.
    The black-box function can represent several different structures. See `black_box` for usage examples.

    Parameters
    ----------
    black_box : BlackBox
        The black box to approximate as a MPS.
    initial_points : Optional[Matrix], default=None
        A collection of initial points used to initialize the algorithm.
        If None, the point at origin is used.
    cross_strategy : CrossStrategyPRRLU = CrossStrategyPRRLU()
        A dataclass containing the parameters of the algorithm.

    Returns
    -------
    CrossResults
        A dataclass containing the MPS representation of the black-box function,
        among other useful information.
    """
    if cross_strategy is None:
        cross_strategy = CrossStrategyPRRLU()
    return cross_interpolation(cross_strategy, black_box, initial_points)
