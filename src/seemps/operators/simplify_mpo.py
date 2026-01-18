from __future__ import annotations
from math import isqrt
from . import MPO, MPOList, MPOSum
from ..state import (
    DEFAULT_STRATEGY,
    MPS,
    Strategy,
    Simplification,
    SIMPLIFICATION_STRATEGY,
    NO_TRUNCATION,
    simplify,
)

#: MPO simplification strategy based on canonical forms without truncation.
CANONICALIZE_MPO = NO_TRUNCATION.replace(simplify=Simplification.DO_NOT_SIMPLIFY)

# MPO simplification should not normalize by default
MPO_SIMPLIFICATION_STRATEGY = SIMPLIFICATION_STRATEGY.replace(normalize=False)


def mpo_as_mps(mpo: MPO) -> MPS:
    """Recast MPO as MPS."""
    data = []
    for site in mpo._data:
        bl, i, j, br = site.shape
        data.append(site.reshape(bl, i * j, br))

    return MPS(data)


def mps_as_mpo(
    mps: MPS,
    mpo_strategy: Strategy = DEFAULT_STRATEGY,
) -> MPO:
    """Recast MPS as MPO."""
    data = []
    for site in mps._data:
        bl, p, br = site.shape
        s = isqrt(p)
        if s**2 != p:
            raise ValueError(
                "The physical dimensions of the MPS must be a perfect square"
            )

        data.append(site.reshape(bl, s, s, br))

    return MPO(data, strategy=mpo_strategy)


def simplify_mpo(
    operator: MPO | MPOList | MPOSum,
    strategy: Strategy = MPO_SIMPLIFICATION_STRATEGY,
    direction: int = +1,
    guess: MPO | None = None,
    mpo_strategy: Strategy | None = None,
) -> MPO:
    """Simplify an MPO state transforming it into another one with a smaller bond
    dimension, sweeping until convergence is achieved.

    Parameters
    ----------x
    operator : MPO | MPOList | MPOSum
        MPO to simplify. If given as `MPOList` or `MPOSum`, it is joined to `MPO`
        before the simplification.
    strategy : Strategy, default=SIMPLIFICATION_STRATEGY
        Truncation strategy to use in the simplification routine.
    direction : int, default=1
        Initial direction for the sweeping algorithm.
    guess : MPS, optional
        Guess for the new state, to ease the optimization.
    mpo_strategy : Strategy | None
        Strategy of the resulting MPO (defaults to the one from `operator`)

    Returns
    -------
    MPO
        Approximation O to the operator.
    """
    if isinstance(operator, MPOList) or isinstance(operator, MPOSum):
        operator = operator.join()
    if mpo_strategy is None:
        mpo_strategy = operator.strategy
    mps = mpo_as_mps(operator)
    if guess is None:
        guess_mps = mps
    else:
        guess_mps = mpo_as_mps(guess)
    mps = simplify(mps, strategy, direction, guess_mps)
    return mps_as_mpo(mps, mpo_strategy=mpo_strategy)
