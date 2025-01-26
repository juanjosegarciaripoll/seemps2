from typing import Union
from math import isqrt

from ..operators import MPO, MPOList, MPOSum
from ..state import DEFAULT_STRATEGY, MPS, Strategy
from ..truncate import SIMPLIFICATION_STRATEGY, simplify


def mpo_as_mps(mpo: MPO) -> MPS:
    """Recast MPO as MPS."""
    _, i, j, _ = mpo[0].shape
    return MPS([t.reshape(t.shape[0], i * j, t.shape[-1]) for t in mpo._data])


def mps_as_mpo(
    mps: MPS,
    mpo_strategy: Strategy = DEFAULT_STRATEGY,
) -> MPO:
    """Recast MPS as MPO."""
    _, S, _ = mps[0].shape
    s = isqrt(S)
    if s**2 != S:
        raise ValueError("The physical dimensions of the MPS must be a perfect square")
    return MPO(
        [t.reshape(t.shape[0], s, s, t.shape[-1]) for t in mps._data],
        strategy=mpo_strategy,
    )


def simplify_mpo(
    operator: Union[MPO, MPOList, MPOSum],
    strategy: Strategy = SIMPLIFICATION_STRATEGY,
    direction: int = +1,
    guess: MPS | None = None,
    mpo_strategy: Strategy = DEFAULT_STRATEGY,
) -> MPO:
    """Simplify an MPO state transforming it into another one with a smaller bond
    dimension, sweeping until convergence is achieved.

    Parameters
    ----------
    operator : Union[MPO, MPOList, MPOSum]
        MPO to simplify. If given as `MPOList` or `MPOSum`, it is joined to `MPO`
        before the simplification.
    strategy : Strategy, default=SIMPLIFICATION_STRATEGY
        Truncation strategy to use in the simplification routine.
    direction : int, default=1
        Initial direction for the sweeping algorithm.
    guess : MPS, optional
        Guess for the new state, to ease the optimization.
    mpo_strategy : Strategy, default=DEFAULT_STRATEGY
        Strategy of the resulting MPO.

    Returns
    -------
    MPO
        Approximation O to the operator.
    """
    if isinstance(operator, MPOList) or isinstance(operator, MPOSum):
        operator = operator.join()
    mps = simplify(mpo_as_mps(operator), strategy, direction, guess)
    return mps_as_mpo(mps, mpo_strategy=mpo_strategy)
