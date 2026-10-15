from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Literal

from ...typing import Matrix, Vector
from .black_box import BlackBox
from .kernels import _combine_indices, prrlu
from .cross import (
    CrossInterpolation,
    CrossResults,
    CrossStrategy,
    cross_interpolation,
)

# TODO: Refactor


def _prrlu_core_left(L: Matrix, I: Vector) -> Matrix:
    """Return L @ inv(L[I]), whose rows I form the identity matrix."""
    return np.linalg.solve(L[I].T, L.T).T


def _prrlu_core_right(U: Matrix, J: Vector) -> Matrix:
    """Return inv(U[:, J]) @ U, whose columns J form the identity matrix."""
    pivots = U[:, J]
    if not np.any(np.diag(pivots)):  # Factorization of a zero matrix
        return np.ones_like(U)
    return np.linalg.solve(pivots, U)


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
        has at most ``full_pivot_threshold`` entries. On binary bonds it also
        uses full search when the initial rook sample spans the whole superblock.
    maxiter_rook : int, default=3
        Maximum alternating block-rook moves, plus a final factorization if
        needed to match the sweep direction.
    full_pivot_threshold : int, default=4096
        Maximum superblock size for size-based automatic full pivoting. The
        binary rook-coverage rule can select full search above this size. Use
        ``pivot_search="rook"`` to force rook search in all cases.

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
        super().__init__(
            black_box, initial_points, cache_samples=strategy.cache_samples
        )
        self.strategy = strategy
        # Prefixes/suffixes can coincide even for distinct initial points.
        if initial_points is not None and len(initial_points) > 1:
            self.I_l = [np.unique(a, axis=0) for a in self.I_l]
            self.I_g = [np.unique(a, axis=0) for a in self.I_g]
        if self.sites == 1:
            self.mps[0] = self.sample_fiber(0)

    def _sample_pivots(
        self,
        rows: Matrix,
        columns: Matrix,
        selected: Vector,
        cache: dict[int, Vector],
        by_column: bool,
    ) -> Matrix:
        """Sample selected rows or columns, reusing values within one rook search."""
        missing = [p for p in selected if p not in cache]
        if missing:
            if by_column:
                values = self.sample(rows, columns[missing]).reshape(len(rows), -1).T
            else:
                values = self.sample(rows[missing], columns).reshape(-1, len(columns))
            cache.update(zip(missing, values))
            if len(missing) == len(selected):
                return values.T if by_column else values
        values = np.array([cache[p] for p in selected])
        return values.T if by_column else values

    def _factor(self, rows: Matrix, columns: Matrix, k: int, forward: bool):
        """Choose pivots by full or block-rook PRRLU and return the carry block.

        A forward sweep needs factors spanning all rows and carries A[I, :].
        A backward sweep needs factors spanning all columns and carries A[:, J].
        """
        strategy = self.strategy
        max_rank = strategy.range_max_bonds[1]
        tolerance = strategy.pivot_tol
        full = strategy.pivot_search == "full" or (
            strategy.pivot_search == "auto"
            and max(self.black_box.physical_dimensions[k : k + 2]) <= 16
            and len(rows) * len(columns) <= strategy.full_pivot_threshold
        )
        if full:
            A = self.sample(rows, columns).reshape(len(rows), len(columns))
            if strategy.error_relative:
                tolerance *= float(np.max(np.abs(A)))
            I, J, L, U = prrlu(A, tolerance, max_rank)
            return I, J, L, U, A[I, :] if forward else A[:, J]

        # Seed block rook with surviving old pivots and equally many new ones.
        old = self.I_g[k] if forward else self.I_l[k + 1]
        candidates = columns if forward else rows
        position = {point: i for i, point in enumerate(map(tuple, candidates.tolist()))}
        selected = np.array(
            [position[p] for p in map(tuple, old.tolist()) if p in position], dtype=int
        )
        if (
            strategy.pivot_search == "auto"
            and all(d == 2 for d in self.black_box.physical_dimensions[k : k + 2])
            and len(selected) + max(1, len(old)) >= len(candidates)
        ):
            A = self.sample(rows, columns).reshape(len(rows), len(columns))
            if strategy.error_relative:
                tolerance *= float(np.max(np.abs(A)))
            I, J, L, U = prrlu(A, tolerance, max_rank)
            return I, J, L, U, A[I, :] if forward else A[:, J]

        available_mask = np.ones(len(candidates), dtype=bool)
        available_mask[selected] = False
        available = np.flatnonzero(available_mask)
        extra = strategy.rng.choice(
            available, min(max(1, len(old)), len(available)), replace=False
        )
        selected = np.concatenate((selected, extra))

        row_cache: dict[int, Vector] = {}
        column_cache: dict[int, Vector] = {}
        scale = 0.0
        previous = None
        search_columns = forward
        iteration = 0
        finished = False
        while True:
            # Factor A[:, selected] or A[selected, :], then restore global pivots.
            A = self._sample_pivots(
                rows,
                columns,
                selected,
                column_cache if search_columns else row_cache,
                search_columns,
            )
            if strategy.error_relative:
                scale = max(scale, float(np.max(np.abs(A))))
                tolerance = strategy.pivot_tol * scale
            I, J, L, U = prrlu(A, tolerance, max_rank)
            if search_columns:
                J = selected[J]
            else:
                I = selected[I]

            iteration += 1
            finished = (
                finished
                or iteration >= strategy.maxiter_rook
                or (
                    previous is not None
                    and np.array_equal(I, previous[0])
                    and np.array_equal(J, previous[1])
                )
            )
            # Take one extra step if the factors do not match the sweep direction.
            if finished and search_columns == forward:
                break
            previous = I, J
            selected = I if search_columns else J
            search_columns = not search_columns

        carry = self._sample_pivots(
            rows,
            columns,
            I if forward else J,
            row_cache if forward else column_cache,
            not forward,
        )
        return I, J, L, U, carry

    def update(self, k: int, left_to_right: bool) -> None:
        rows = _combine_indices(self.I_l[k], self.I_s[k])
        columns = _combine_indices(self.I_s[k + 1], self.I_g[k + 1])
        I, J, L, U, carry = self._factor(rows, columns, k, left_to_right)
        rank = len(I)
        r_l, s1 = len(self.I_l[k]), len(self.I_s[k])
        s2, r_g = len(self.I_s[k + 1]), len(self.I_g[k + 1])
        if left_to_right:
            core = _prrlu_core_left(L, I)
            self.mps[k] = core.reshape(r_l, s1, rank)
            self.mps[k + 1] = carry.reshape(rank, s2, r_g)
        else:
            core = _prrlu_core_right(U, J)
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
