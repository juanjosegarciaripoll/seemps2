import numpy as np
from dataclasses import dataclass

from .cross import (
    BlackBox,
    CrossResults,
    CrossInterpolation,
    CrossStrategy,
    maxvol_square,
    _check_convergence,
)
from ...state import Strategy, DEFAULT_TOLERANCE
from ...state._contractions import _contract_last_and_first
from ...state.schmidt import _destructive_svd
from ...state.core import destructively_truncate_vector
from ...truncate import SIMPLIFICATION_STRATEGY
from ...tools import make_logger

DEFAULT_CROSS_STRATEGY = SIMPLIFICATION_STRATEGY.replace(
    normalize=False,
    tolerance=DEFAULT_TOLERANCE**2,
    simplification_tolerance=DEFAULT_TOLERANCE**2,
)


@dataclass
class CrossStrategyDMRG(CrossStrategy):
    maxvol_tol: float = 1.1
    maxvol_maxiter: int = 100
    strategy: Strategy = DEFAULT_CROSS_STRATEGY
    """
    Dataclass containing the parameters for the rectangular maxvol-based TCI.
    The common parameters are documented in the base `CrossStrategy` class.

    Parameters
    ----------
    maxvol_tol : float, default = 1.1
        Sensibility for the square maxvol decomposition.
    maxvol_maxiter : int, default = 100
        Maximum number of iterations for the square maxvol decomposition.
    strategy : Strategy, default = DEFAULT_CROSS_STRATEGY
        Simplification strategy used at the truncation of Schmidt values
        at each SVD split of the DMRG superblocks.
    """


class CrossInterpolationDMRG(CrossInterpolation):
    def __init__(self, black_box: BlackBox, initial_point: np.ndarray):
        super().__init__(black_box, initial_point)

    def sample_superblock(self, k: int) -> np.ndarray:
        i_l, i_g = self.I_l[k], self.I_g[k + 1]
        i_s1, i_s2 = self.I_s[k], self.I_s[k + 1]
        mps_indices = self.combine_indices(i_l, i_s1, i_s2, i_g)
        return self.black_box[mps_indices].reshape(
            (len(i_l), len(i_s1), len(i_s2), len(i_g))
        )


def cross_dmrg(
    black_box: BlackBox,
    cross_strategy: CrossStrategyDMRG = CrossStrategyDMRG(),
) -> CrossResults:
    """
    Computes the MPS representation of a black-box function using the tensor cross-approximation (TCI)
    algorithm based on two-site optimizations in a DMRG-like manner.

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
    cross = CrossInterpolationDMRG(black_box, initial_point)
    converged = False
    with make_logger(2) as logger:
        for i in range(cross_strategy.maxiter):
            # Forward sweep
            for k in range(cross.sites - 1):
                _update_dmrg(cross, k, True, cross_strategy)
            if converged := _check_convergence(cross, i, cross_strategy, logger):
                break
            # Backward sweep
            for k in reversed(range(cross.sites - 1)):
                _update_dmrg(cross, k, False, cross_strategy)
            if converged := _check_convergence(cross, i, cross_strategy, logger):
                break
        if not converged:
            logger("Maximum number of TT-Cross iterations reached")
    return CrossResults(mps=cross.mps, evals=black_box.evals)


def _update_dmrg(
    cross: CrossInterpolationDMRG,
    k: int,
    forward: bool,
    cross_strategy: CrossStrategyDMRG,
) -> None:
    superblock = cross.sample_superblock(k)
    r_l, s1, s2, r_g = superblock.shape
    A = superblock.reshape(r_l * s1, s2 * r_g)
    ## SVD
    U, S, V = _destructive_svd(A)
    destructively_truncate_vector(S, cross_strategy.strategy)
    r = S.size
    U, S, V = U[:, :r], np.diag(S), V[:r, :]
    ##
    if forward:
        C = U.reshape(r_l * s1, r)
        Q, T = np.linalg.qr(C)
        I, G = maxvol_square(
            Q, cross_strategy.maxvol_maxiter, cross_strategy.maxvol_tol
        )
        cross.I_l[k + 1] = cross.combine_indices(cross.I_l[k], cross.I_s[k])[I]
        cross.mps[k] = G.reshape(r_l, s1, r)
        if k == cross.sites - 2:
            cross.mps[k] = _contract_last_and_first(cross.mps[k], Q[I] @ T)
            cross.mps[k + 1] = (S @ V).reshape(r, s2, r_g)
    else:
        R = V.reshape(r, s2 * r_g)
        Q, T = np.linalg.qr(R.T)
        I, G = maxvol_square(
            Q, cross_strategy.maxvol_maxiter, cross_strategy.maxvol_tol
        )
        cross.mps[k + 1] = (G.T).reshape(r, s2, r_g)
        cross.I_g[k] = cross.combine_indices(cross.I_s[k + 1], cross.I_g[k + 1])[I]
        if k == 0:
            cross.mps[k] = (U @ S).reshape(r_l, s1, r)
            cross.mps[k + 1] = _contract_last_and_first((Q[I] @ T).T, cross.mps[k + 1])
