from typing import Optional

from ..operators import MPO
from ..state import DEFAULT_STRATEGY, MPS, Strategy
from ..truncate import SIMPLIFICATION_STRATEGY, simplify


def mpo_as_mps(mpo):
    """Recast MPO as MPS."""
    _, i, j, _ = mpo[0].shape
    return MPS([t.reshape(t.shape[0], i * j, t.shape[-1]) for t in mpo._data])


def simplify_mpo(
    operator: MPO,
    strategy: Strategy = SIMPLIFICATION_STRATEGY,
    direction: int = +1,
    guess: Optional[MPS] = None,
    mpo_strategy: Strategy = DEFAULT_STRATEGY,
) -> MPO:
    """Simplify an MPS state transforming it into another one with a smaller bond
    dimension, sweeping until convergence is achieved.

    Parameters
    ----------
    operator : MPO
    MPO to approximate.
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
    _, i, j, _ = operator[0].shape
    mps = simplify(mpo_as_mps(operator), strategy, direction, guess)
    [t.reshape(t.shape[0], i, j, t.shape[-1]) for t in mps._data]
    return MPO(
        [t.reshape(t.shape[0], i, j, t.shape[-1]) for t in mps._data],
        strategy=mpo_strategy,
    )
