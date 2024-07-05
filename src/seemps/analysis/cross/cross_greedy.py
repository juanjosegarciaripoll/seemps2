import numpy as np
import scipy.linalg
from typing import TypeVar, Union, Optional, Callable
from dataclasses import dataclass

from .cross import (
    CrossInterpolation,
    CrossResults,
    CrossStrategy,
    BlackBox,
    _check_convergence,
)
from ..sampling import random_mps_indices
from ...state import MPS
from ...state._contractions import _contract_last_and_first
from ...tools import make_logger, Logger


@dataclass
class CrossStrategyGreedy(CrossStrategy):
    tol_pivot: float = 1e-10
    partial: bool = True
    maxiter_partial: int = 5
    points_partial: int = 10
    """
    Dataclass containing parameters for TCI with greedy pivot updates.
    Supplements the base `CrossStrategy` class. 

    Parameters
    ----------
    tol_pivot : float, default=1e-12
        Minimum allowable error for a pivot, excluding those below this threshold. 
        The algorithm halts when the maximum pivot error across all sites falls below this limit.
    partial : bool, default=True
        Whether to use a row-column alternating partial search strategy to find pivots in the superblock. 
        If False, performs a 'full search' that uses more function evaluations (O(chi) vs. O(chi^2)) but
        can introduce potentially smaller errors.
    maxiter_partial : int, default=5
        Number of row-column iterations in each partial search.
    points_partial : int, default=10
        Number of initial random points used to initialize each partial search.
    """


def cross_greedy(
    black_box: BlackBox,
    cross_strategy: CrossStrategyGreedy = CrossStrategyGreedy(),
    initial_points: Optional[np.ndarray] = None,
    callback: Optional[Callable] = None,
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
    cross = CrossInterpolationGreedy(black_box, initial_points)

    if cross_strategy.partial == True:
        update_method = _update_partial_search
    else:
        update_method = _update_full_search

    pivot_errors = np.zeros((black_box.sites - 1,))
    converged = False
    callback_output = []
    with make_logger(2) as logger:
        for i in range(cross_strategy.maxiter):
            # Forward sweep
            direction = True
            for k in range(cross.sites - 1):
                pivot_errors[k] = update_method(cross, k, cross_strategy)
            if callback:
                callback_output.append(callback(cross.mps, logger=logger))
            if converged := (
                _check_convergence(cross, i, cross_strategy, logger)
                or _check_local_convergence(pivot_errors, cross_strategy, logger)
            ):
                break
            # Backward sweep
            direction = False
            for k in reversed(range(cross.sites - 1)):
                pivot_errors[k] = update_method(cross, k, cross_strategy)
            if callback:
                callback_output.append(callback(cross.mps, logger=logger))
            if converged := (
                _check_convergence(cross, i, cross_strategy, logger)
                or _check_local_convergence(pivot_errors, cross_strategy, logger)
            ):
                break
        if not converged:
            logger("Maximum number of TT-Cross iterations reached")
    points = cross.indices_to_points(direction)
    return CrossResults(
        mps=cross.mps,
        points=points,
        evals=black_box.evals,
        callback_output=callback_output,
    )


class CrossInterpolationGreedy(CrossInterpolation):
    def __init__(self, black_box: BlackBox, initial_point: np.ndarray):
        super().__init__(black_box, initial_point)
        self.fibers = [self.sample_fiber(k) for k in range(self.sites)]
        self.Q_factors = []
        self.R_matrices = []
        for fiber in self.fibers[:-1]:
            Q, R = self.fiber_to_QR(fiber)
            self.Q_factors.append(Q)
            self.R_matrices.append(R)

        ## Translate the initial multiindices I_l and I_g to integer indices J_l and J_g
        ## TODO: Refactor
        def get_row_indices(rows, all_rows):
            large_set = {tuple(row): idx for idx, row in enumerate(all_rows)}
            return np.array([large_set[tuple(row)] for row in rows])

        J_l = []
        J_g = []
        for k in range(self.sites - 1):
            i_l = self.combine_indices(self.I_l[k], self.I_s[k])
            J_l.append(get_row_indices(self.I_l[k + 1], i_l))
            i_g = self.combine_indices(self.I_l[k], self.I_s[k])
            J_g.append(get_row_indices(self.I_l[k + 1], i_g))
        self.J_l = [np.array([])] + J_l  # add empty indices to respect convention
        self.J_g = J_g[::-1] + [np.array([])]
        ##

        G_cores = [self.Q_to_G(Q, j_l) for Q, j_l in zip(self.Q_factors, self.J_l[1:])]
        self.mps = MPS(G_cores + [self.fibers[-1]])

    _Index = TypeVar("_Index", bound=Union[np.intp, np.ndarray, slice])

    def sample_superblock(
        self, k: int, j_l: _Index = slice(None), j_g: _Index = slice(None)
    ) -> np.ndarray:
        i_ls = self.combine_indices(self.I_l[k], self.I_s[k])[j_l]
        i_ls = i_ls.reshape(1, -1) if i_ls.ndim == 1 else i_ls  # Prevent collapse to 1D
        i_sg = self.combine_indices(self.I_s[k + 1], self.I_g[k + 1])[j_g]
        i_sg = i_sg.reshape(1, -1) if i_sg.ndim == 1 else i_sg
        mps_indices = self.combine_indices(i_ls, i_sg)
        return self.black_box[mps_indices].reshape((len(i_ls), len(i_sg)))

    def sample_skeleton(
        self, k: int, j_l: _Index = slice(None), j_g: _Index = slice(None)
    ) -> np.ndarray:
        r_l, r_s1, chi = self.mps[k].shape
        chi, r_s2, r_g = self.fibers[k + 1].shape
        G = self.mps[k].reshape(r_l * r_s1, chi)[j_l]
        R = self.fibers[k + 1].reshape(chi, r_s2 * r_g)[:, j_g]
        return _contract_last_and_first(G, R)

    def update_indices(self, k: int, j_l: _Index, j_g: _Index) -> None:
        i_l = self.combine_indices(self.I_l[k], self.I_s[k])[j_l]
        i_g = self.combine_indices(self.I_s[k + 1], self.I_g[k + 1])[j_g]
        self.I_l[k + 1] = np.vstack((self.I_l[k + 1], i_l))
        self.J_l[k + 1] = np.append(self.J_l[k + 1], j_l)  # type: ignore
        self.I_g[k] = np.vstack((self.I_g[k], i_g))
        self.J_g[k] = np.append(self.J_g[k], j_g)  # type: ignore

    def update_tensors(
        self,
        k: int,
        r: np.ndarray,
        c: np.ndarray,
    ) -> None:
        # Update left fiber, Q-factor and MPS site
        r_l, r_s1, chi = self.fibers[k].shape
        C = self.fibers[k].reshape(r_l * r_s1, chi)
        self.fibers[k] = np.hstack((C, c.reshape(-1, 1))).reshape(r_l, r_s1, chi + 1)

        QR_INSERT = False  # TODO: Check why it is unstable
        if QR_INSERT:
            Q = self.Q_factors[k].reshape(r_l * r_s1, chi)
            Q, self.R_matrices[k] = scipy.linalg.qr_insert(
                Q,
                self.R_matrices[k],
                u=c,
                k=Q.shape[1],
                which="col",
                rcond=None,
                check_finite=False,
            )
            self.Q_factors[k] = Q.reshape(r_l, r_s1, chi + 1)
        else:
            self.Q_factors[k], self.R_matrices[k] = self.fiber_to_QR(self.fibers[k])
        self.mps[k] = self.Q_to_G(self.Q_factors[k], self.J_l[k + 1])

        # Update right fiber, Q-factor and MPS site
        chi, r_s2, r_g = self.fibers[k + 1].shape
        R = self.fibers[k + 1].reshape(chi, r_s2 * r_g)
        self.fibers[k + 1] = np.vstack((R, r)).reshape(chi + 1, r_s2, r_g)
        if k < self.sites - 2:
            if QR_INSERT:
                Q = self.Q_factors[k + 1].reshape(chi * r_s2, r_g)
                Q, self.R_matrices[k + 1] = scipy.linalg.qr_insert(
                    Q,
                    self.R_matrices[k + 1],
                    u=r.reshape(-1, Q.shape[1]),
                    k=Q.shape[0],
                    which="row",
                    check_finite=False,
                )
                self.Q_factors[k + 1] = Q.reshape(chi + 1, r_s2, r_g)
            else:
                self.Q_factors[k + 1], self.R_matrices[k + 1] = self.fiber_to_QR(
                    self.fibers[k + 1]
                )
            self.mps[k + 1] = self.Q_to_G(self.Q_factors[k + 1], self.J_l[k + 2])
        else:
            self.mps[k + 1] = self.fibers[k + 1]

    @staticmethod
    def fiber_to_QR(fiber: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Performs the QR decomposition of a fiber."""
        r_l, r_s, r_g = fiber.shape
        Q, R = scipy.linalg.qr(  # type: ignore
            fiber.reshape(r_l * r_s, r_g), mode="economic", check_finite=False
        )
        Q_factor = Q.reshape(r_l, r_s, r_g)  # type: ignore
        return Q_factor, R

    @staticmethod
    def Q_to_G(Q_factor: np.ndarray, j_l: np.ndarray) -> np.ndarray:
        """Transforms a Q-factor into a MPS tensor core G."""
        r_l, r_s, r_g = Q_factor.shape
        Q = Q_factor.reshape(r_l * r_s, r_g)
        P = scipy.linalg.inv(Q[j_l], check_finite=False)
        G = _contract_last_and_first(Q, P)
        return G.reshape(r_l, r_s, r_g)


def _update_full_search(
    cross: CrossInterpolationGreedy,
    k: int,
    cross_strategy: CrossStrategyGreedy,
) -> float:
    max_pivots = cross.black_box.base ** (1 + min(k, cross.sites - (k + 2)))
    if len(cross.I_g[k]) >= max_pivots or len(cross.I_l[k + 1]) >= max_pivots:
        return 0

    A = cross.sample_superblock(k)
    B = cross.sample_skeleton(k)

    error_function = lambda A, B: np.abs(A - B)
    diff = error_function(A, B)
    j_l, j_g = np.unravel_index(np.argmax(diff), A.shape)
    pivot_error = diff[j_l, j_g]

    if pivot_error >= cross_strategy.tol_pivot:
        cross.update_indices(k, j_l=j_l, j_g=j_g)
        cross.update_tensors(k, r=A[j_l, :], c=A[:, j_g])

    return pivot_error


def _update_partial_search(
    cross: CrossInterpolationGreedy,
    k: int,
    cross_strategy: CrossStrategyGreedy,
) -> float:
    max_pivots = cross.black_box.base ** (1 + min(k, cross.sites - (k + 2)))
    if len(cross.I_g[k]) >= max_pivots or len(cross.I_l[k + 1]) >= max_pivots:
        return 0

    j_l_random = cross_strategy.rng.integers(
        low=0,
        high=len(cross.I_l[k]) * len(cross.I_s[k]),
        size=cross_strategy.points_partial,
    )
    j_g_random = cross_strategy.rng.integers(
        low=0,
        high=len(cross.I_s[k + 1]) * len(cross.I_g[k + 1]),
        size=cross_strategy.points_partial,
    )
    A_random = cross.sample_superblock(k, j_l=j_l_random, j_g=j_g_random)
    B_random = cross.sample_skeleton(k, j_l=j_l_random, j_g=j_g_random)

    error_function = lambda A, B: np.abs(A - B)
    diff = error_function(A_random, B_random)
    i, j = np.unravel_index(np.argmax(diff), A_random.shape)
    j_l, j_g = j_l_random[i], j_g_random[j]

    for iter in range(cross_strategy.maxiter_partial):
        # Traverse column residual
        c_A = cross.sample_superblock(k, j_g=j_g).reshape(-1)
        c_B = cross.sample_skeleton(k, j_g=j_g)
        new_j_l = np.argmax(error_function(c_A, c_B))
        if new_j_l == j_l and iter > 0:
            break
        j_l = new_j_l

        # Traverse row residual
        r_A = cross.sample_superblock(k, j_l=j_l).reshape(-1)
        r_B = cross.sample_skeleton(k, j_l=j_l)
        new_j_g = np.argmax(error_function(r_A, r_B))
        if new_j_g == j_g:
            break
        j_g = new_j_g
    pivot_error = error_function(c_A[j_l], c_B[j_l])

    if pivot_error >= cross_strategy.tol_pivot:
        cross.update_indices(k, j_l=j_l, j_g=j_g)
        cross.update_tensors(k, r=r_A, c=c_A)

    return pivot_error


def _check_local_convergence(
    pivot_errors: np.ndarray,
    cross_strategy: CrossStrategyGreedy,
    logger: Logger,
) -> bool:
    max_pivot_error = np.max(pivot_errors)
    if logger:
        logger(f"Max. pivot error={max_pivot_error}")
    if max_pivot_error < cross_strategy.tol_pivot:
        logger(f"State converged within tolerance {cross_strategy.tol_pivot}")
        return True
    return False
