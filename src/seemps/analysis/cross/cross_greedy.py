from __future__ import annotations
import numpy as np
import scipy.linalg
from dataclasses import dataclass
from ...state import MPS
from ...cython import _contract_last_and_first
from ...typing import Vector, Matrix, Tensor3
from .black_box import BlackBox
from .kernels import combine_indices
from .cross import (
    CrossStrategy,
    CrossInterpolation,
    CrossResults,
    IndexSlice,
    IndexMatrix,
    IndexVector,
    cross_interpolation,
)

# TODO: Remove from SeeMPS (much more complex and slower than other TCI variants)


@dataclass
class CrossStrategyGreedy(CrossStrategy):
    tol_pivot: float = 1e-10
    maxiter: int = 5
    initial_points: int = 10
    """
    Dataclass containing the parameters for greedy TCI.
    The common parameters are documented in the base `CrossStrategy` class.

    Parameters
    ----------
    tol_pivot : float, default=1e-10
        Minimum allowable error for a pivot, relative to the largest entry of the
        sampled row and column, excluding those below this threshold.
        The algorithm halts when the maximum pivot error across all sites falls below this limit.
        A local roundoff floor also prevents adding numerical noise as a pivot.
    maxiter : int, default=5
        Number of row-column iterations in each partial search.
    initial_points : int, default=10
        Number of initial random points used to initialize each partial search.
    """

    def make_interpolator(
        self, black_box: BlackBox, initial_points: Matrix | None = None
    ) -> CrossInterpolation:
        return CrossInterpolationGreedy(self, black_box, initial_points)


class CrossInterpolationGreedy(CrossInterpolation):
    strategy: CrossStrategyGreedy
    fibers: list[Tensor3]
    mps: MPS
    J_l: list
    J_g: list

    def __init__(
        self,
        strategy: CrossStrategyGreedy,
        black_box: BlackBox,
        initial_points: Matrix | None,
    ):
        super().__init__(
            black_box,
            initial_points,
            two_site_algorithm=True,
            cache_evals=strategy.cache_evals,
        )
        self.strategy = strategy
        if strategy.maxiter < 1 or strategy.initial_points < 1:
            raise ValueError("maxiter and initial_points must be positive")

        self.fibers = [self.sample_fiber(k) for k in range(self.sites)]

        # Translate initial multi-indices I_l and I_g to integer indices J_l and J_g.
        def get_row_indices(rows: IndexMatrix, all_rows: IndexMatrix) -> IndexMatrix:
            large_set = {tuple(row): idx for idx, row in enumerate(all_rows)}
            return np.array([large_set[tuple(row)] for row in rows])

        J_l = []
        J_g = []
        for k in range(self.sites - 1):
            i_l = combine_indices(self.I_l[k], self.I_s[k])
            i_g = combine_indices(self.I_s[k + 1], self.I_g[k + 1])
            J_l.append(get_row_indices(self.I_l[k + 1], i_l))
            J_g.append(get_row_indices(self.I_g[k], i_g))

        self.J_l = [np.array([])] + J_l  # add empty indices to respect convention
        self.J_g = J_g + [np.array([])]

        mps_cores = [
            self.fiber_to_core(fiber, j_l)
            for fiber, j_l in zip(self.fibers[:-1], self.J_l[1:])
        ]
        self.mps = MPS(mps_cores + [self.fibers[-1]])

    def sample_superblock(
        self, k: int, j_l: IndexSlice = slice(None), j_g: IndexSlice = slice(None)
    ) -> Matrix:
        i_ls = combine_indices(self.I_l[k], self.I_s[k])[j_l]
        i_ls = i_ls.reshape(1, -1) if i_ls.ndim == 1 else i_ls  # Prevent collapse to 1D
        i_sg = combine_indices(self.I_s[k + 1], self.I_g[k + 1])[j_g]
        i_sg = i_sg.reshape(1, -1) if i_sg.ndim == 1 else i_sg
        return self.sample(i_ls, i_sg).reshape((len(i_ls), len(i_sg)))

    def sample_skeleton(
        self, k: int, j_l: IndexSlice = slice(None), j_g: IndexSlice = slice(None)
    ) -> Matrix:
        r_l, r_s1, chi = self.mps[k].shape
        chi, r_s2, r_g = self.fibers[k + 1].shape
        G = self.mps[k].reshape(r_l * r_s1, chi)[j_l]
        R = self.fibers[k + 1].reshape(chi, r_s2 * r_g)[:, j_g]
        return _contract_last_and_first(G, R)

    def update_indices(
        self, k: int, j_l: np.intp | IndexVector, j_g: np.intp | IndexVector
    ) -> None:
        i_l = combine_indices(self.I_l[k], self.I_s[k])[j_l]
        i_g = combine_indices(self.I_s[k + 1], self.I_g[k + 1])[j_g]
        self.I_l[k + 1] = np.vstack((self.I_l[k + 1], i_l))
        self.J_l[k + 1] = np.append(self.J_l[k + 1], j_l)
        self.I_g[k] = np.vstack((self.I_g[k], i_g))
        self.J_g[k] = np.append(self.J_g[k], j_g)
        if k > 0:
            # Columns at bond k-1 flatten (physical index, right suffix).
            # Appending a suffix changes their stride, not their multi-indices.
            old_rank = len(self.I_g[k]) - 1
            self.J_g[k - 1] += self.J_g[k - 1] // old_rank

    def update_tensors(self, k: int, row: Vector, column: Vector) -> None:
        # Update left fiber using the column vector
        fiber_1 = self.fibers[k]
        r_l, r_s1, chi = fiber_1.shape
        C = fiber_1.reshape(r_l * r_s1, chi)
        fiber_1_new = np.hstack((C, column.reshape(-1, 1))).reshape(r_l, r_s1, chi + 1)
        self.fibers[k] = fiber_1_new

        self.mps[k] = self.fiber_to_core(fiber_1_new, self.J_l[k + 1])

        # Update right fiber using the row vector
        fiber_2 = self.fibers[k + 1]
        chi, r_s2, r_g = fiber_2.shape
        R = fiber_2.reshape(chi, r_s2 * r_g)
        fiber_2_new = np.vstack((R, row)).reshape(chi + 1, r_s2, r_g)
        self.fibers[k + 1] = fiber_2_new

        if k < self.sites - 2:
            self.mps[k + 1] = self.fiber_to_core(fiber_2_new, self.J_l[k + 2])
        else:
            self.mps[k + 1] = self.fibers[k + 1]

    @staticmethod
    def fiber_to_core(fiber: Tensor3, row_indices: Vector) -> Tensor3:
        """Form a cross core using row-equilibrated QR and a right solve."""
        r_l, r_s, r_g = fiber.shape
        F = fiber.reshape(r_l * r_s, r_g)
        # QR is normwise stable, but may lose tiny selected rows. Equilibrate
        # them first: F = D Q R, hence G = D Q inv(Q[J]) inv(D[J]).
        scale = np.max(np.abs(F), axis=1)
        scale[scale == 0] = 1
        Q, _ = scipy.linalg.qr(F / scale[:, None], mode="economic")
        G = scipy.linalg.solve(Q[row_indices].T, Q.T).T
        G *= scale[:, None] / scale[row_indices]
        G[row_indices] = np.eye(r_g)  # Enforce the exact interpolation identity.
        return G.reshape(r_l, r_s, r_g)

    def update(self, k: int, left_to_right: bool) -> None:
        cross_strategy = self.strategy
        n_rows = len(self.I_l[k]) * len(self.I_s[k])
        n_cols = len(self.I_s[k + 1]) * len(self.I_g[k + 1])
        if len(self.I_g[k]) >= min(n_rows, n_cols):
            return

        # The exact residual vanishes on the cross. Never promote roundoff
        # there to a pivot: duplicate rows/columns make the cross singular.
        rows = np.delete(np.arange(n_rows), self.J_l[k + 1])
        cols = np.delete(np.arange(n_cols), self.J_g[k])
        j_l_random = cross_strategy.rng.choice(rows, size=cross_strategy.initial_points)
        j_g_random = cross_strategy.rng.choice(cols, size=cross_strategy.initial_points)
        A_random = self.sample_superblock(k, j_l=j_l_random, j_g=j_g_random)
        B_random = self.sample_skeleton(k, j_l=j_l_random, j_g=j_g_random)
        diff = np.abs(A_random - B_random)
        i, j = np.unravel_index(np.argmax(diff), A_random.shape)
        j_l, j_g = j_l_random[i], j_g_random[j]

        c_A = c_B = r_A = r_B = np.empty(0)
        for _ in range(cross_strategy.maxiter):
            # Traverse column residual
            c_A = self.sample_superblock(k, j_g=j_g).reshape(-1)
            c_B = self.sample_skeleton(k, j_g=j_g)
            j_l = rows[np.argmax(np.abs(c_A - c_B)[rows])]

            # Traverse row residual
            r_A = self.sample_superblock(k, j_l=j_l).reshape(-1)
            r_B = self.sample_skeleton(k, j_l=j_l)
            new_j_g = cols[np.argmax(np.abs(r_A - r_B)[cols])]
            if new_j_g == j_g:
                break
            j_g = new_j_g
        else:
            # The last row search changed j_g. Cache its column, not the old one.
            c_A = self.sample_superblock(k, j_g=j_g).reshape(-1)
            c_B = self.sample_skeleton(k, j_g=j_g)

        pivot_error = np.abs(c_A[j_l] - c_B[j_l])
        # Tolerances below cancellation noise must not inflate rank.
        scale = max(np.max(np.abs(a)) for a in (c_A, c_B, r_A, r_B))
        roundoff = 10 * np.finfo(c_B.dtype).eps * len(self.I_g[k])
        if pivot_error > max(cross_strategy.tol_pivot, roundoff) * scale:
            self.update_indices(k, j_l=j_l, j_g=j_g)
            self.update_tensors(k, row=r_A, column=c_A)


def cross_greedy(
    black_box: BlackBox,
    initial_points: Matrix | None = None,
    cross_strategy: CrossStrategyGreedy = CrossStrategyGreedy(),
) -> CrossResults:
    """
    Computes the MPS representation of a black-box function using the tensor cross-interpolation (TCI)
    algorithm based on two-site optimizations following greedy updates of the pivot matrices.
    The black-box function can represent several different structures. See `black_box` for usage examples.

    Parameters
    ----------
    black_box : BlackBox
        The black box to approximate as a MPS.
    initial_points : np.ndarray, optional
        A collection of initial points used to initialize the algorithm.
        If None, an initial point at the origin is used.
    cross_strategy : CrossStrategy, default=CrossStrategy()
        A dataclass containing the parameters of the algorithm.

    Returns
    -------
    CrossResults
        A dataclass containing the MPS representation of the black-box function,
        among other useful information.
    """
    return cross_interpolation(cross_strategy, black_box, initial_points)
