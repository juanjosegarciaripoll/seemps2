from __future__ import annotations

import numpy as np
from scipy.linalg import solve_triangular
from dataclasses import dataclass
from typing import Literal, cast

from ...typing import Matrix
from .black_box import BlackBox
from .cross import CrossInterpolation, CrossResults, CrossStrategy, cross_interpolation


@dataclass
class CrossStrategyPRRLU(CrossStrategy):
    """Parameters for reset-mode, rank-adaptive LU cross interpolation.

    Parameters
    ----------
    pivot_tol : float, default=1e-12
        Absolute local Schur-complement tolerance, or relative to the largest
        sampled entry when ``error_relative`` is True. This is distinct from
        the global sampled-error tolerance ``tol``.
    pivot_search : {"auto", "full", "rook"}, default="auto"
        Full pivoting or block-rook search. Auto uses full search
        when both physical dimensions are at most sixteen and the superblock
        has at most ``full_pivot_threshold`` entries.
    maxiter_rook : int, default=3
        Maximum alternating block-rook moves, plus a final factorization if
        needed to match the sweep direction.
    full_pivot_threshold : int, default=4096
        Maximum superblock size for automatic full pivoting.

    Notes
    -----
    Both searches reset pivots and can reduce ranks. Rook searches and the
    global error samples are heuristic: sparse features can be missed. Supply
    representative ``initial_points`` for disconnected nonzero regions.
    """

    pivot_tol: float = 1e-12
    pivot_search: Literal["auto", "full", "rook"] = "auto"
    maxiter_rook: int = 3
    full_pivot_threshold: int = 4096

    def __post_init__(self) -> None:
        super().__post_init__()
        if not np.isfinite(self.pivot_tol) or self.pivot_tol < 0:
            raise ValueError("pivot_tol must be finite and nonnegative")
        if self.pivot_search not in ("auto", "full", "rook"):
            raise ValueError("pivot_search must be 'auto', 'full', or 'rook'")
        if self.maxiter_rook < 1 or self.full_pivot_threshold < 0:
            raise ValueError("Invalid rook iteration count or full-pivot threshold")
        if (
            self.range_max_bonds[1] < self.range_max_bonds[0]
            or self.range_iters[1] < self.range_iters[0]
        ):
            raise ValueError("Iteration and bond ranges must be ordered")

    def make_interpolator(
        self, black_box: BlackBox, initial_points: Matrix | None = None
    ) -> CrossInterpolationPRRLU:
        return CrossInterpolationPRRLU(self, black_box, initial_points)


def _prrlu(
    A: Matrix, tolerance: float, max_rank: int
) -> tuple[Matrix, Matrix, Matrix, Matrix]:
    residual = np.array(A, dtype=np.result_type(A, np.float64), copy=True)
    if not np.all(np.isfinite(residual)):
        raise ValueError("PRRLU requires finite black-box values")
    m, n = residual.shape
    limit = min(m, n, max_rank)
    L, U = np.empty((m, limit), residual.dtype), np.empty((limit, n), residual.dtype)
    I, J = np.empty(limit, dtype=int), np.empty(limit, dtype=int)
    rank = 0
    for p in range(limit):
        i, j = np.unravel_index(np.abs(residual).argmax(), (m, n))
        pivot = residual[i, j]
        if pivot == 0 or (p > 0 and abs(pivot) <= tolerance):
            break
        I[p], J[p] = i, j
        L[:, p], U[p, :] = residual[:, j] / pivot, residual[i, :]
        residual -= L[:, p, None] * U[p, None, :]
        residual[i, :], residual[:, j] = 0, 0
        rank = p + 1
    if rank == 0:
        return np.array([0]), np.array([0]), np.ones((m, 1)), np.zeros((1, n), A.dtype)
    return I[:rank], J[:rank], L[:, :rank], U[:rank]


class CrossInterpolationPRRLU(CrossInterpolation):
    strategy: CrossStrategyPRRLU
    I_l: list[Matrix]
    I_g: list[Matrix]

    def __init__(
        self,
        strategy: CrossStrategyPRRLU,
        black_box: BlackBox,
        initial_points: Matrix | None = None,
    ):
        if not black_box.physical_dimensions or min(black_box.physical_dimensions) < 1:
            raise ValueError("At least one positive physical dimension is required")
        if initial_points is not None:
            points = np.atleast_2d(initial_points)
            if (
                points.shape[1] != len(black_box.physical_dimensions)
                or len(points) == 0
                or not np.issubdtype(points.dtype, np.integer)
                or np.any(points < 0)
                or np.any(points >= black_box.physical_dimensions)
            ):
                raise ValueError(
                    "initial_points must contain valid integer tensor indices"
                )
            initial_points = points
        super().__init__(black_box, initial_points)
        self.strategy = strategy
        # Prefixes/suffixes can coincide even for distinct initial points.
        if initial_points is not None and len(initial_points) > 1:
            self.I_l = [np.unique(a, axis=0) for a in self.I_l]
            self.I_g = [np.unique(a, axis=0) for a in self.I_g]
        if self.sites == 1:
            self.mps[0] = self.sample_fiber(0)

    def _factor(self, rows: Matrix, columns: Matrix, k: int, forward: bool):
        strategy = self.strategy
        max_rank = strategy.range_max_bonds[1]
        scale = 0.0
        row_cache, column_cache = {}, {}

        def evaluate(
            i: Matrix | list[int] | slice, j: Matrix | list[int] | slice
        ) -> Matrix:
            indices = self.combine_indices(rows[i], columns[j])
            return self.black_box[indices].reshape(len(rows[i]), len(columns[j]))

        def sample(i: Matrix | slice, j: Matrix | slice) -> Matrix:
            # Cache rows/columns locally, without a dense superblock cache
            # or reuse of samples across updates.
            if isinstance(i, slice) and isinstance(j, slice):
                return evaluate(i, j)
            by_column = isinstance(i, slice)
            selected, cache = (j, column_cache) if by_column else (i, row_cache)
            selected = cast(Matrix, selected)
            missing = [p for p in selected if p not in cache]
            if missing:
                values = (
                    evaluate(slice(None), missing).T
                    if by_column
                    else evaluate(missing, slice(None))
                )
                cache.update(zip(missing, values))
            values = np.array([cache[p] for p in selected])
            return values.T if by_column else values

        def factor(A: Matrix):
            nonlocal scale
            if strategy.error_relative:
                scale = max(scale, float(np.max(np.abs(A))))
            tol = strategy.pivot_tol * (scale if strategy.error_relative else 1.0)
            return _prrlu(A, tol, max_rank)

        full = strategy.pivot_search == "full" or (
            strategy.pivot_search == "auto"
            and max(self.black_box.physical_dimensions[k : k + 2]) <= 16
            and len(rows) * len(columns) <= strategy.full_pivot_threshold
        )
        if full:
            A = sample(slice(None), slice(None))
            I, J, L, U = factor(A)
            return I, J, L, U, A[I, :] if forward else A[:, J]

        # Seed block rook with surviving old pivots and equally many new ones.
        old = self.I_g[k] if forward else self.I_l[k + 1]
        candidates = columns if forward else rows
        lookup = {tuple(row): i for i, row in enumerate(candidates)}
        selected = np.array(
            [lookup[tuple(row)] for row in old if tuple(row) in lookup], dtype=int
        )
        available = np.setdiff1d(np.arange(len(candidates)), selected)
        extra = strategy.rng.choice(
            available, min(max(1, len(old)), len(available)), replace=False
        )
        selected = np.concatenate((selected, extra))
        previous = None
        search_columns = forward
        iteration = 0
        while True:
            A = (
                sample(slice(None), selected)
                if search_columns
                else sample(selected, slice(None))
            )
            I, J, L, U = factor(A)
            if search_columns:
                J = selected[J]
            else:
                I = selected[I]
            iteration += 1
            if iteration >= strategy.maxiter_rook or (
                previous is not None
                and np.array_equal(I, previous[0])
                and np.array_equal(J, previous[1])
            ):
                break
            previous = I, J
            selected = I if search_columns else J
            search_columns = not search_columns
        if search_columns != forward:
            A = sample(slice(None), J) if forward else sample(I, slice(None))
            i, j, L, U = factor(A)
            I, J = (i, J[j]) if forward else (I[i], j)
        carry = sample(I, slice(None)) if forward else sample(slice(None), J)
        return I, J, L, U, carry

    def update(self, k: int, left_to_right: bool) -> None:
        rows = self.combine_indices(self.I_l[k], self.I_s[k])
        columns = self.combine_indices(self.I_s[k + 1], self.I_g[k + 1])
        I, J, L, U, carry = self._factor(rows, columns, k, left_to_right)
        rank = len(I)
        r_l, s1 = len(self.I_l[k]), len(self.I_s[k])
        s2, r_g = len(self.I_s[k + 1]), len(self.I_g[k + 1])
        if left_to_right:
            core = solve_triangular(
                L[I], L.T, lower=True, unit_diagonal=True, trans="T", check_finite=False
            ).T
            self.mps[k] = core.reshape(r_l, s1, rank)
            self.mps[k + 1] = carry.reshape(rank, s2, r_g)
        else:
            diagonal = U[np.arange(rank), J]
            # A zero local block still needs a nonsingular interpolation gauge.
            if np.all(diagonal == 0):
                core = np.ones((rank, len(columns)), dtype=U.dtype)
            else:
                U = U / diagonal[:, None]
                core = solve_triangular(
                    U[:, J], U, unit_diagonal=True, check_finite=False
                )
            self.mps[k] = carry.reshape(r_l, s1, rank)
            self.mps[k + 1] = core.reshape(rank, s2, r_g)
        self.I_l[k + 1], self.I_g[k] = rows[I], columns[J]


def cross_prrlu(
    black_box: BlackBox,
    initial_points: Matrix | None = None,
    cross_strategy: CrossStrategyPRRLU | None = None,
) -> CrossResults:
    """Interpolate a black box with reset-mode PRRLU (see CrossStrategyPRRLU)."""
    if cross_strategy is None:
        cross_strategy = CrossStrategyPRRLU()
    return cross_interpolation(cross_strategy, black_box, initial_points)
