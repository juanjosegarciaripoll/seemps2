from __future__ import annotations

import numpy as np
import scipy.linalg
from dataclasses import dataclass
from typing import Callable

from ...state import Strategy, DEFAULT_TOLERANCE, SIMPLIFICATION_STRATEGY
from ...cython import destructively_truncate_vector
from ...typing import Matrix
from ...tools import make_logger
from .black_box import BlackBox
from .cross import (
    CrossStrategy,
    CrossInterpolation,
    CrossError,
    CrossResults,
    check_tci_convergence,
    maxvol_square,
)


DEFAULT_CROSS_DMRG_STRATEGY = SIMPLIFICATION_STRATEGY.replace(
    normalize=False,
    tolerance=DEFAULT_TOLERANCE**2,
    simplification_tolerance=DEFAULT_TOLERANCE**2,
)


@dataclass
class CrossStrategyDMRG(CrossStrategy):
    strategy: Strategy = DEFAULT_CROSS_DMRG_STRATEGY
    tol_maxvol_square: float = 1.05
    maxiter_maxvol: int = 10
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

    @property
    def algorithm(self) -> Callable:
        return cross_dmrg


def cross_dmrg(
    black_box: BlackBox,
    cross_strategy: CrossStrategyDMRG = CrossStrategyDMRG(),
    initial_points: Matrix | None = None,
) -> CrossResults:
    """
    Computes the MPS representation of a black-box function using the tensor cross-approximation (TCI)
    algorithm based on two-site optimizations in a DMRG-like manner.
    The black-box function can represent several different structures. See `black_box` for usage examples.

    Parameters
    ----------
    black_box : BlackBox
        The black box to approximate as a MPS.
    cross_strategy : CrossStrategy, default=CrossStrategyDMRG()
        A dataclass containing the parameters of the algorithm.
    initial_points : Matrix | None, default=None
        A collection of initial points used to initialize the algorithm.
        If None, an initial point at the origin is used.

    Returns
    -------
    CrossResults
        A dataclass containing the MPS representation of the black-box function,
        among other useful information.
    """
    cross = CrossInterpolation(black_box, initial_points)
    error_calculator = CrossError(cross_strategy)

    converged = False
    with make_logger(1) as logger:
        results = CrossResults(cross.mps)
        for i in range(cross_strategy.range_iters[1] // 2):
            # Left-to-right half sweep
            for k in range(cross.sites - 1):
                _update_cross(cross, k, True, cross_strategy)

            results.update(
                cross.mps,
                error_calculator.sample_error(cross),
                cross.mps.bond_dimensions(),
                cross.black_box.evals,
            )
            if converged := check_tci_convergence(
                logger, 2 * i + 1, results, cross_strategy
            ):
                break

            # Right-to-left half sweep
            for k in reversed(range(cross.sites - 1)):
                _update_cross(cross, k, False, cross_strategy)

            results.update(
                cross.mps,
                error_calculator.sample_error(cross),
                cross.mps.bond_dimensions(),
                cross.black_box.evals,
            )
            if converged := check_tci_convergence(
                logger, 2 * i + 2, results, cross_strategy
            ):
                break

        if not converged:
            logger("Maximum number of iterations reached")

    return results


def _update_cross(
    cross: CrossInterpolation,
    k: int,
    left_to_right: bool,
    cross_strategy: CrossStrategyDMRG,
) -> None:
    superblock = cross.sample_superblock(k)
    r_l, s1, s2, r_g = superblock.shape
    A = superblock.reshape(r_l * s1, s2 * r_g)

    ## Non-destructive SVD
    U, S, V = scipy.linalg.svd(A, check_finite=False)
    destructively_truncate_vector(S, cross_strategy.strategy)
    r = S.size
    U, S, V = U[:, :r], np.diag(S), V[:r, :]
    ##
    r = S.shape[0]

    if left_to_right:
        if k < cross.sites - 2:
            C = U.reshape(r_l * s1, r)
            Q, _ = scipy.linalg.qr(
                C, mode="economic", overwrite_a=True, check_finite=False
            )  # type: ignore
            I, G = maxvol_square(
                Q,
                cross_strategy.maxiter_maxvol,
                cross_strategy.tol_maxvol_square,  # type: ignore
            )
            cross.I_l[k + 1] = cross.combine_indices(cross.I_l[k], cross.I_s[k])[I]
            cross.mps[k] = G.reshape(r_l, s1, r)
        else:
            cross.mps[k] = U.reshape(r_l, s1, r)
            cross.mps[k + 1] = (S @ V).reshape(r, s2, r_g)

    else:
        if k > 0:
            R = V.reshape(r, s2 * r_g)
            Q, _ = scipy.linalg.qr(
                R.T, mode="economic", overwrite_a=True, check_finite=False
            )  # type: ignore
            I, G = maxvol_square(
                Q,
                cross_strategy.maxiter_maxvol,
                cross_strategy.tol_maxvol_square,  # type: ignore
            )
            cross.I_g[k] = cross.combine_indices(cross.I_s[k + 1], cross.I_g[k + 1])[I]
            cross.mps[k + 1] = (G.T).reshape(r, s2, r_g)
        else:
            cross.mps[k] = (U @ S).reshape(r_l, s1, r)
            cross.mps[k + 1] = V.reshape(r, s2, r_g)
