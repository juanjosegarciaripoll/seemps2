from ...state import MPS, MPSSum, Strategy, DEFAULT_STRATEGY
from ...operators import MPOList
from ...qft import qft_mpo
from ..operators import p_to_n_mpo, p_mpo
from ..mesh import QuantizedInterval, IntervalTuple


def fourier_derivative_mpo(
    order: int,
    interval: QuantizedInterval | IntervalTuple,
    strategy: Strategy = DEFAULT_STRATEGY,
) -> MPOList:
    """Nth-order derivative operator via Quantum Fourier Transform.

    Compute the `order`-th spatial derivative operator that can be applied on
    any function encoded as quantized tensor train, and defined on a uniform
    grid of length `L`.

    Parameters
    ----------
    order : int
        Order of the derivative
    interval: QuantizedInterval | IntervalTuple
        The interval over which the function is defined.

    Returns
    -------
    MPOList
        Operator that implements the derivative.
    """
    if isinstance(interval, tuple):
        interval = QuantizedInterval(*interval)
    L = interval.qubits
    dx = interval.step
    forward_qft = qft_mpo(L, -1, strategy)
    if order == 1:
        P_op = 1j * p_mpo(L, dx)
    else:
        P_op = ((1j) ** order) * p_to_n_mpo(L, dx, order)
    backward_qft = qft_mpo(L, +1, strategy)
    return (backward_qft @ P_op).reverse() @ forward_qft


def fourier_derivative(
    psi: MPS | MPSSum, order: int, interval: QuantizedInterval | IntervalTuple
) -> MPS | MPSSum:
    """Nth-order derivative via Quantum Fourier Transform.

    Compute the `order`-th spatial derivative of a quantum state `psi`
    defined on a uniform grid of length `L`, using the QFT and momentum-space
    multiplication.

    Parameters
    ----------
    psi : MPS or MPSSum
        Quantum state in position representation
    order : int
        Order of the derivative
    interval: QuantizedInterval | IntervalTuple
        The interval over which the function is defined.

    Returns
    -------
    d_psi : MPS or MPSSum
        Quantum state encoding the derivative
    """
    return fourier_derivative_mpo(order, interval).apply(psi)
