from typing import Optional, Union
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


# TODO: As opposed to MPS, the MPO class does not have an error attribute to keep track
# of the simplification errors
def simplify_mpo(
    operator: Union[MPO, MPOList, MPOSum],
    strategy: Strategy = SIMPLIFICATION_STRATEGY,
    direction: int = +1,
    guess: Optional[MPS] = None,
    mpo_strategy: Strategy = DEFAULT_STRATEGY,
) -> MPO:
    """Simplify an MPS state transforming it into another one with a smaller bond
    dimension, sweeping until convergence is achieved.

    Parameters
    ----------
    operator : Union[MPO, MPOList, MPOSum]
        Operator to approximate.
    strategy : Strategy
        Truncation strategy. Defaults to `SIMPLIFICATION_STRATEGY`.
    direction : { +1, -1 }
        Initial direction for the sweeping algorithm. Defaults to +1.
    guess : MPS
        A guess for the new state, to ease the optimization. Defaults to None.
    mpo_strategy : Strategy
        Strategy of the resulting MPO. Defaults to `DEFAULT_STRATEGY`.

    Returns
    -------
    MPO
    Approximation O to the operator.
    """
    if isinstance(operator, MPOList) or isinstance(operator, MPOSum):
        operator = operator.join()
    mps = simplify(mpo_as_mps(operator), strategy, direction, guess)
    return mps_as_mpo(mps, mpo_strategy=mpo_strategy)
