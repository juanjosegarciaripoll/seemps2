import numpy as np
import scipy.linalg
from dataclasses import dataclass
from collections import defaultdict
from time import perf_counter
from typing import Callable, Any

from ...state import MPS
from ...cython import _contract_last_and_first
from ...typing import Vector, Matrix, Tensor3
from ...tools import make_logger
from .black_box import BlackBox
from .cross import (
    CrossStrategy,
    CrossInterpolation,
    CrossError,
    CrossResults,
    check_convergence,
    IndexSlice,
    IndexMatrix,
    IndexVector,
)


@dataclass
class CrossStrategyGreedy(CrossStrategy):
    pivot_tol: float = 1e-10
    maxiter: int = 5
    initial_points: int = 10
    """
    Dataclass containing the parameters for greedy TCI.
    The common parameters are documented in the base `CrossStrategy` class.

    Parameters
    ----------
    pivot_tol : float, default=1e-10
        Minimum allowable error for a pivot, excluding those below this threshold.
        The algorithm halts when the maximum pivot error across all sites falls below this limit.
    maxiter_partial : int, default=5
        Number of row-column iterations in each partial search.
    initial_points : int, default=10
        Number of initial random points used to initialize each partial search.
    """

    @property
    def algorithm(self) -> Callable:
        return cross_greedy


class CrossInterpolationGreedy(CrossInterpolation):
    mps: MPS
    fibers: list[Tensor3]
    list_Q3: list[Tensor3]
    list_R: list[Matrix]
    J_l: list
    J_g: list

    def __init__(self, black_box: BlackBox, initial_points: Matrix | None):
        super().__init__(black_box, initial_points)

        self.fibers = [self.sample_fiber(k) for k in range(self.sites)]
        self.list_Q3 = []  # Q factors of the QR decomposition of a fiber.
        self.list_R = []  # R factors of the QR decomposition of a fiber.

        for fiber in self.fibers[:-1]:
            Q3, R = self.fiber_to_Q3R(fiber)
            self.list_Q3.append(Q3)
            self.list_R.append(R)

        # Translate initial multi-indices I_l and I_g to integer indices J_l and J_g.
        ###
        ## TODO: Refactor this block of code
        ## WARNING: This looks superfishy
        def get_row_indices(rows: IndexMatrix, all_rows: IndexMatrix) -> IndexMatrix:
            large_set = {tuple(row): idx for idx, row in enumerate(all_rows)}
            return np.array([large_set[tuple(row)] for row in rows])

        J_l = []
        J_g = []
        for k in range(self.sites - 1):
            # WARNING: Is this correct? i_l and i_g and the values
            # we add to J_l and J_g have the same value
            i_l = self.combine_indices(self.I_l[k], self.I_s[k])
            i_g = self.combine_indices(self.I_l[k], self.I_s[k])
            J_l.append(get_row_indices(self.I_l[k + 1], i_l))
            J_g.append(get_row_indices(self.I_l[k + 1], i_g))

        self.J_l = [np.array([])] + J_l  # add empty indices to respect convention
        self.J_g = J_g[::-1] + [np.array([])]
        ###

        mps_cores = [
            self.Q3_to_core(Q3, j_l) for Q3, j_l in zip(self.list_Q3, self.J_l[1:])
        ]
        self.mps = MPS(mps_cores + [self.fibers[-1]])

    def sample_superblock(
        self, k: int, j_l: IndexSlice = slice(None), j_g: IndexSlice = slice(None)
    ) -> Matrix:
        i_ls = self.combine_indices(self.I_l[k], self.I_s[k])[j_l]
        i_ls = i_ls.reshape(1, -1) if i_ls.ndim == 1 else i_ls  # Prevent collapse to 1D
        i_sg = self.combine_indices(self.I_s[k + 1], self.I_g[k + 1])[j_g]
        i_sg = i_sg.reshape(1, -1) if i_sg.ndim == 1 else i_sg
        mps_indices = self.combine_indices(i_ls, i_sg)
        return self.black_box[mps_indices].reshape((len(i_ls), len(i_sg)))

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
        i_l = self.combine_indices(self.I_l[k], self.I_s[k])[j_l]
        i_g = self.combine_indices(self.I_s[k + 1], self.I_g[k + 1])[j_g]
        self.I_l[k + 1] = np.vstack((self.I_l[k + 1], i_l))
        self.J_l[k + 1] = np.append(self.J_l[k + 1], j_l)
        self.I_g[k] = np.vstack((self.I_g[k], i_g))
        self.J_g[k] = np.append(self.J_g[k], j_g)

    def update_tensors(self, k: int, row: Vector, column: Vector) -> None:
        # Update left fiber using the column vector
        fiber_1 = self.fibers[k]
        r_l, r_s1, chi = fiber_1.shape
        C = fiber_1.reshape(r_l * r_s1, chi)
        fiber_1_new = np.hstack((C, column.reshape(-1, 1))).reshape(r_l, r_s1, chi + 1)
        self.fibers[k] = fiber_1_new

        # Update left Q3, R and MPS core
        self.list_Q3[k], self.list_R[k] = self.fiber_to_Q3R(fiber_1_new)
        self.mps[k] = self.Q3_to_core(self.list_Q3[k], self.J_l[k + 1])

        # Update right fiber using the row vector
        fiber_2 = self.fibers[k + 1]
        chi, r_s2, r_g = fiber_2.shape
        R = fiber_2.reshape(chi, r_s2 * r_g)
        fiber_2_new = np.vstack((R, row)).reshape(chi + 1, r_s2, r_g)
        self.fibers[k + 1] = fiber_2_new

        # Update right Q3, R and MPS core
        if k < self.sites - 2:
            self.list_Q3[k + 1], self.list_R[k + 1] = self.fiber_to_Q3R(fiber_2_new)
            self.mps[k + 1] = self.Q3_to_core(self.list_Q3[k + 1], self.J_l[k + 2])
        else:
            self.mps[k + 1] = self.fibers[k + 1]

    @staticmethod
    def fiber_to_Q3R(fiber: Tensor3) -> tuple[Tensor3, Matrix]:
        """Performs the QR decomposition of a fiber rank-3 tensor."""
        r_l, r_s, r_g = fiber.shape
        Q, R = scipy.linalg.qr(fiber.reshape(r_l * r_s, r_g), mode="economic")  # type: ignore
        Q3 = Q.reshape(r_l, r_s, r_g)  # type: ignore
        return Q3, R

    @staticmethod
    def Q3_to_core(Q3: Tensor3, row_indices: Vector) -> Tensor3:
        """Transforms the Q rank-3 tensor into a MPS core."""
        r_l, r_s, r_g = Q3.shape
        Q = Q3.reshape(r_l * r_s, r_g)
        P = scipy.linalg.inv(Q[row_indices])
        G = _contract_last_and_first(Q, P)
        return G.reshape(r_l, r_s, r_g)


def cross_greedy(
    black_box: BlackBox,
    cross_strategy: CrossStrategyGreedy = CrossStrategyGreedy(),
    initial_points: Matrix | None = None,
) -> CrossResults:
    """
    Computes the MPS representation of a black-box function using the tensor cross-approximation (TCI)
    algorithm based on two-site optimizations following greedy updates of the pivot matrices.
    The black-box function can represent several different structures. See `black_box` for usage examples.

    Parameters
    ----------
    black_box : BlackBox
        The black box to approximate as a MPS.
    cross_strategy : CrossStrategy, default=CrossStrategy()
        A dataclass containing the parameters of the algorithm.
    initial_points : np.ndarray, optional
        A collection of initial points used to initialize the algorithm.
        If None, an initial point at the origin is used.

    Returns
    -------
    CrossResults
        A dataclass containing the MPS representation of the black-box function,
        among other useful information.
    """
    cross = CrossInterpolationGreedy(black_box, initial_points)
    error_calculator = CrossError(cross_strategy)

    converged = False
    trajectories: defaultdict[str, list[Any]] = defaultdict(list)
    for i in range(cross_strategy.max_iters // 2):
        # Left-to-right half sweep
        tick = perf_counter()
        for k in range(cross.sites - 1):
            _update_cross(cross, k, cross_strategy)
        time_ltr = perf_counter() - tick

        # Update trajectories
        trajectories["errors"].append(error_calculator.sample_error(cross))
        trajectories["bonds"].append(cross.mps.bond_dimensions())
        trajectories["times"].append(time_ltr)
        trajectories["evals"].append(cross.black_box.evals)

        # Evaluate convergence
        if converged := check_convergence(2 * i + 1, trajectories, cross_strategy):
            break

        # Right-to-left half sweep
        tick = perf_counter()
        for k in reversed(range(cross.sites - 1)):
            _update_cross(cross, k, cross_strategy)
        time_rtl = perf_counter() - tick

        # Update trajectories
        trajectories["errors"].append(error_calculator.sample_error(cross))
        trajectories["bonds"].append(cross.mps.bond_dimensions())
        trajectories["times"].append(time_rtl)
        trajectories["evals"].append(cross.black_box.evals)

        # Evaluate convergence
        if converged := check_convergence(2 * i + 2, trajectories, cross_strategy):
            break

    if not converged:
        with make_logger(2) as logger:
            logger("Maximum number of iterations reached")

    return CrossResults(
        mps=cross.mps,
        errors=np.array(trajectories["errors"]),
        bonds=np.array(trajectories["bonds"]),
        times=np.array(trajectories["times"]),
        evals=np.array(trajectories["evals"]),
    )


def _update_cross(
    cross: CrossInterpolationGreedy,
    k: int,
    cross_strategy: CrossStrategyGreedy,
) -> None:
    max_pivots = cross.black_box.physical_dimensions[k] ** (
        1 + min(k, cross.sites - (k + 2))
    )
    if len(cross.I_g[k]) >= max_pivots or len(cross.I_l[k + 1]) >= max_pivots:
        return

    j_l_random = cross_strategy.rng.integers(
        low=0,
        high=len(cross.I_l[k]) * len(cross.I_s[k]),
        size=cross_strategy.initial_points,
    )
    j_g_random = cross_strategy.rng.integers(
        low=0,
        high=len(cross.I_s[k + 1]) * len(cross.I_g[k + 1]),
        size=cross_strategy.initial_points,
    )
    A_random = cross.sample_superblock(k, j_l=j_l_random, j_g=j_g_random)
    B_random = cross.sample_skeleton(k, j_l=j_l_random, j_g=j_g_random)
    diff = np.abs(A_random - B_random)
    i, j = np.unravel_index(np.argmax(diff), A_random.shape)
    j_l, j_g = j_l_random[i], j_g_random[j]

    c_A = c_B = r_A = r_B = np.empty(0)
    for iter in range(cross_strategy.max_iters):
        # Traverse column residual
        c_A = cross.sample_superblock(k, j_g=j_g).reshape(-1)
        c_B = cross.sample_skeleton(k, j_g=j_g)
        new_j_l = np.argmax(np.abs(c_A - c_B))
        if new_j_l == j_l and iter > 0:
            break
        j_l = new_j_l

        # Traverse row residual
        r_A = cross.sample_superblock(k, j_l=j_l).reshape(-1)
        r_B = cross.sample_skeleton(k, j_l=j_l)
        new_j_g = np.argmax(np.abs(r_A - r_B))
        if new_j_g == j_g:
            break
        j_g = new_j_g

    pivot_error = np.abs(c_A[j_l] - c_B[j_l])
    if pivot_error >= cross_strategy.pivot_tol:
        cross.update_indices(k, j_l=j_l, j_g=j_g)
        cross.update_tensors(k, row=r_A, column=c_A)
