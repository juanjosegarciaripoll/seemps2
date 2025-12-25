from math import isqrt
from . import MPO, MPOList, MPOSum
from ..state import DEFAULT_STRATEGY, MPS, Strategy, SIMPLIFICATION_STRATEGY, simplify


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
    strategy: Strategy = SIMPLIFICATION_STRATEGY,
    direction: int = +1,
    guess: MPS | None = None,
    mpo_strategy: Strategy = DEFAULT_STRATEGY,
) -> MPO:
    """Simplify an MPO state transforming it into another one with a smaller bond
    dimension, sweeping until convergence is achieved.

    Parameters
    ----------
    operator : MPO | MPOList | MPOSum
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
