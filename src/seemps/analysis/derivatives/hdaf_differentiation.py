from ...state import Strategy, DEFAULT_STRATEGY
from ...operators import MPO
from ...typing import Float
from ..mesh import QuantizedInterval, IntervalTuple
from ..hdaf import hdaf_mpo


def hdaf_derivative_mpo(
    order: int,
    interval: QuantizedInterval | IntervalTuple,
    M: int = 10,
    s0: Float | None = None,
    periodic: bool = True,
    strategy: Strategy = DEFAULT_STRATEGY,
) -> MPO:
    """
    Constructs a Matrix Product Operator (MPO) of Hermite Distributed
    Approximating Functionals (HDAFs). The operator may approximate the
    identity, a derivative or the free propagator, depending on the values
    of the `derivative` and `time` parameters.

    Parameters
    ----------
    order : int
        Order of the derivative to approximate
    interval : QuantizedInterval | IntervalTuple
        The interval over which the function is defined.
    periodic : bool, default=True
        Whether the grid follows perioidic boundary conditions.
    M : int
        The order of the highest Hermite polynomial (must be an even integer).
        Defaults to 10.
    s0 : Float | None, default=None
        The width of the HDAF Gaussian weight. If not provided, a suitable
        width will be computed based on `M` and `dx`.
    strategy : Strategy, default=DEFAULT_STRATEGY
        The strategy for the returned MPO. Values of the HDAF below the
        simplification tolerance of the strategy will be discarded.

    Returns
    -------
    mpo: MPO
        The HDAF approximation to an operator specified by the input parameters.
    """
    if isinstance(interval, tuple):
        interval = QuantizedInterval(*interval)
    return hdaf_mpo(
        interval.qubits,
        interval.step,
        M,
        s0,
        derivative=order,
        periodic=periodic,
        strategy=strategy,
    )
