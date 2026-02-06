from __future__ import annotations
import numpy as np
import scipy.linalg
from dataclasses import dataclass
from ...state import Strategy, DEFAULT_TOLERANCE, SIMPLIFICATION_STRATEGY
from ...cython import destructively_truncate_vector
from ...typing import Matrix
from .black_box import BlackBox
from .cross import (
    CrossStrategy,
    CrossInterpolation,
    CrossResults,
    maxvol_square,
    cross_interpolation,
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

    def make_interpolator(
        self, black_box: BlackBox, initial_points: Matrix | None = None
    ) -> CrossInterpolation:
        return CrossInterpolationDMRG(self, black_box, initial_points)


class CrossInterpolationDMRG(CrossInterpolation):
    strategy: CrossStrategyDMRG

    def __init__(
        self,
        strategy: CrossStrategyDMRG,
        black_box: BlackBox,
        initial_points: Matrix | None = None,
    ):
        super().__init__(black_box, initial_points, two_site_algorithm=True)
        self.strategy = strategy

    def update(self, k: int, left_to_right: bool) -> None:
        superblock = self.sample_superblock(k)
        cross_strategy = self.strategy
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
            if k < self.sites - 2:
                C = U.reshape(r_l * s1, r)
                Q, _ = scipy.linalg.qr(
                    C, mode="economic", overwrite_a=True, check_finite=False
                )  # type: ignore
                I, G = maxvol_square(
                    Q,
                    cross_strategy.maxiter_maxvol,
                    cross_strategy.tol_maxvol_square,  # type: ignore
                )
                self.I_l[k + 1] = self.combine_indices(self.I_l[k], self.I_s[k])[I]
                self.mps[k] = G.reshape(r_l, s1, r)
            else:
                self.mps[k] = U.reshape(r_l, s1, r)
                self.mps[k + 1] = (S @ V).reshape(r, s2, r_g)

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
                self.I_g[k] = self.combine_indices(self.I_s[k + 1], self.I_g[k + 1])[I]
                self.mps[k + 1] = (G.T).reshape(r, s2, r_g)
            else:
                self.mps[k] = (U @ S).reshape(r_l, s1, r)
                self.mps[k + 1] = V.reshape(r, s2, r_g)


def cross_dmrg(
    black_box: BlackBox,
    initial_points: Matrix | None = None,
    cross_strategy: CrossStrategyDMRG = CrossStrategyDMRG(),
) -> CrossResults:
    """
    Computes the MPS representation of a black-box function using the tensor cross-approximation (TCI)
    algorithm based on two-site optimizations in a DMRG-like manner.
    The black-box function can represent several different structures. See `black_box` for usage examples.

    Parameters
    ----------
    black_box : BlackBox
        The black box to approximate as a MPS.
    initial_points : Matrix | None, default=None
        A collection of initial points used to initialize the algorithm.
        If None, an initial point at the origin is used.
    cross_strategy : CrossStrategy, default=CrossStrategyDMRG()
        A dataclass containing the parameters of the algorithm.

    Returns
    -------
    CrossResults
        A dataclass containing the MPS representation of the black-box function,
        among other useful information.
    """
    return cross_interpolation(cross_strategy, black_box, initial_points)
