import numpy as np
import scipy.linalg  # type: ignore
from dataclasses import dataclass
from typing import Optional, Callable

from .cross import (
    BlackBox,
    CrossResults,
    CrossInterpolation,
    CrossStrategy,
    maxvol_square,
    _check_convergence,
)
from ..sampling import random_mps_indices
from ...state import Strategy, DEFAULT_TOLERANCE
from ...state.schmidt import svd
from ...state.core import destructively_truncate_vector
from ...truncate import SIMPLIFICATION_STRATEGY
from ...tools import make_logger

DEFAULT_CROSS_STRATEGY = SIMPLIFICATION_STRATEGY.replace(
    normalize=False,
    tolerance=DEFAULT_TOLERANCE**2,
    simplification_tolerance=DEFAULT_TOLERANCE**2,
)

# TODO: Implement local error evaluation


@dataclass
class CrossStrategyDMRG(CrossStrategy):
    strategy: Strategy = DEFAULT_CROSS_STRATEGY
    tol_maxvol_square: float = 1.05
    maxiter_maxvol_square: int = 10
    """
    Dataclass containing the parameters for the DMRG-based TCI.
    The common parameters are documented in the base `CrossStrategy` class.

    Parameters
    ----------
    strategy : Strategy, default=DEFAULT_CROSS_STRATEGY
        Simplification strategy used at the truncation of Schmidt values
        at each SVD split of the DMRG superblocks.
    tol_maxvol_square : float, default=1.05
        Sensibility for the square maxvol decomposition.
    maxiter_maxvol_square : int, default=10
        Maximum number of iterations for the square maxvol decomposition.
    """


def cross_dmrg(
    black_box: BlackBox,
    cross_strategy: CrossStrategyDMRG = CrossStrategyDMRG(),
    initial_points: Optional[np.ndarray] = None,
    callback: Optional[Callable] = None,
) -> CrossResults:
    """
    Computes the MPS representation of a black-box function using the tensor cross-approximation (TCI)
    algorithm based on two-site optimizations in a DMRG-like manner.
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

    cross = CrossInterpolationDMRG(black_box, initial_points)
    converged = False
    callback_output = []
    with make_logger(2) as logger:
        for i in range(cross_strategy.maxiter):
            # Forward sweep
            direction = True
            for k in range(cross.sites - 1):
                _update_dmrg(cross, k, direction, cross_strategy)
            if callback:
                callback_output.append(callback(cross.mps, logger=logger))
            if converged := _check_convergence(cross, i, cross_strategy, logger):
                break
            # Backward sweep
            direction = False
            for k in reversed(range(cross.sites - 1)):
                _update_dmrg(cross, k, direction, cross_strategy)
            if callback:
                callback_output.append(callback(cross.mps, logger=logger))
            if converged := _check_convergence(cross, i, cross_strategy, logger):
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


def _update_dmrg(
    cross: CrossInterpolationDMRG,
    k: int,
    forward: bool,
    cross_strategy: CrossStrategyDMRG,
) -> None:
    superblock = cross.sample_superblock(k)
    r_l, s1, s2, r_g = superblock.shape
    A = superblock.reshape(r_l * s1, s2 * r_g)
    ## Non-destructive SVD
    U, S, V = svd(A, check_finite=False)
    destructively_truncate_vector(S, cross_strategy.strategy)
    r = S.size
    U, S, V = U[:, :r], np.diag(S), V[:r, :]
    ##
    if forward:
        if k < cross.sites - 2:
            C = U.reshape(r_l * s1, r)
            Q, _ = scipy.linalg.qr(C, mode="economic", overwrite_a=True, check_finite=False)  # type: ignore
            I, G = maxvol_square(
                Q, cross_strategy.maxiter_maxvol_square, cross_strategy.tol_maxvol_square  # type: ignore
            )
            cross.I_l[k + 1] = cross.combine_indices(cross.I_l[k], cross.I_s[k])[I]
            cross.mps[k] = G.reshape(r_l, s1, r)
        else:
            cross.mps[k] = U.reshape(r_l, s1, r)
            cross.mps[k + 1] = (S @ V).reshape(r, s2, r_g)
    else:
        if k > 0:
            R = V.reshape(r, s2 * r_g)
            Q, _ = scipy.linalg.qr(  # type: ignore
                R.T, mode="economic", overwrite_a=True, check_finite=False
            )
            I, G = maxvol_square(
                Q, cross_strategy.maxiter_maxvol_square, cross_strategy.tol_maxvol_square  # type: ignore
            )
            cross.I_g[k] = cross.combine_indices(cross.I_s[k + 1], cross.I_g[k + 1])[I]
            cross.mps[k + 1] = (G.T).reshape(r, s2, r_g)
        else:
            cross.mps[k] = (U @ S).reshape(r_l, s1, r)
            cross.mps[k + 1] = V.reshape(r, s2, r_g)
