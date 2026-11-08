from __future__ import annotations

import numpy as np
import scipy.linalg
from dataclasses import dataclass
from ...state import Strategy, SIMPLIFICATION_STRATEGY
from ...cython import destructively_truncate_vector
from ...typing import Matrix
from .black_box import BlackBox
from .kernels import combine_indices, maxvol_square, prrlu
from .cross import (
    CrossStrategy,
    CrossInterpolation,
    CrossResults,
    cross_interpolation,
)


DEFAULT_CROSS_DMRG_STRATEGY = SIMPLIFICATION_STRATEGY.replace(
    normalize=False,
    tolerance=(1e-10) ** 2,
    simplification_tolerance=(1e-10) ** 2,
)


@dataclass
class CrossStrategyDMRG(CrossStrategy):
    strategy: Strategy = DEFAULT_CROSS_DMRG_STRATEGY
    tol_maxvol_square: float = 1.05
    maxiter_maxvol: int = 10
    compressed_svd: bool = True
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
    compressed_svd : bool, default=True
        Use a faster approximate SVD for numerically low-rank TT matrices.
        Disable for full-rank matrices.
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
        super().__init__(
            black_box,
            initial_points,
            two_site_algorithm=True,
            cache_evals=strategy.cache_evals,
        )
        self.strategy = strategy

    def update(self, k: int, left_to_right: bool) -> None:
        superblock = self.sample_superblock(k)
        cross_strategy = self.strategy
        r_l, s1, s2, r_g = superblock.shape
        A = superblock.reshape(r_l * s1, s2 * r_g)

        # Compressed SVD (much faster for larger matrices)
        if cross_strategy.compressed_svd:
            svd_tolerance = cross_strategy.strategy.get_tolerance()
            prrlu_tolerance = (
                1e-2 * np.sqrt(svd_tolerance / A.size) * scipy.linalg.norm(A)
            )
            _, _, L, R = prrlu(A, prrlu_tolerance, min(A.shape))
            QL, RL = scipy.linalg.qr(L, mode="economic", check_finite=False)
            QR, RR = scipy.linalg.qr(R.T, mode="economic", check_finite=False)
            U, S, V = scipy.linalg.svd(
                RL @ RR.T, full_matrices=False, check_finite=False
            )
            U, V = QL @ U, V @ QR.T

        # Standard SVD
        else:
            U, S, V = scipy.linalg.svd(A, full_matrices=False, check_finite=False)

        destructively_truncate_vector(S, cross_strategy.strategy)
        r = S.size
        U, V = U[:, :r], V[:r]

        if left_to_right:
            if k < self.sites - 2:
                I, G = maxvol_square(
                    U,
                    cross_strategy.maxiter_maxvol,
                    cross_strategy.tol_maxvol_square,  # type: ignore
                )
                self.I_l[k + 1] = combine_indices(self.I_l[k], self.I_s[k])[I]
                self.mps[k] = G.reshape(r_l, s1, r)
            else:
                self.mps[k] = U.reshape(r_l, s1, r)
                self.mps[k + 1] = (S[:, None] * V).reshape(r, s2, r_g)

        else:
            if k > 0:
                I, G = maxvol_square(
                    V.T,
                    cross_strategy.maxiter_maxvol,
                    cross_strategy.tol_maxvol_square,  # type: ignore
                )
                self.I_g[k] = combine_indices(self.I_s[k + 1], self.I_g[k + 1])[I]
                self.mps[k + 1] = (G.T).reshape(r, s2, r_g)
            else:
                self.mps[k] = (U * S).reshape(r_l, s1, r)
                self.mps[k + 1] = V.reshape(r, s2, r_g)


def cross_dmrg(
    black_box: BlackBox,
    initial_points: Matrix | None = None,
    cross_strategy: CrossStrategyDMRG = CrossStrategyDMRG(),
) -> CrossResults:
    """
    Computes the MPS representation of a black-box function using the tensor cross-interpolation (TCI)
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
