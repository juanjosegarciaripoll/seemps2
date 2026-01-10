from seemps.state import MPS, MPSSum, Strategy, DEFAULT_STRATEGY
from seemps.operators import MPOList
from seemps.qft import qft_mpo
from seemps.analysis.operators import p_to_n_mpo, p_mpo


def fourier_derivative_mpo(
    L: int, order: int, dx: float = 1.0, strategy: Strategy = DEFAULT_STRATEGY
) -> MPOList:
    """Nth-order derivative operator via Quantum Fourier Transform.

    Compute the `order`-th spatial derivative operator that can be applied on
    any function encoded as quantized tensor train, and defined on a uniform
    grid of length `L`.

    Parameters
    ----------
    L : int
        Length of the quantum register
    order : int
        Order of the derivative

    Returns
    -------
    MPOList
        Operator that implements the derivative.
    """
    forward_qft = qft_mpo(L, -1, strategy)
    if order == 1:
        P_op = 1j * p_mpo(L, dx)
    else:
        P_op = ((1j) ** order) * p_to_n_mpo(L, dx, order)
    backward_qft = qft_mpo(L, +1, strategy)
    return (backward_qft @ P_op).reverse() @ forward_qft


def fourier_derivative(psi: MPS | MPSSum, order: int, L: float) -> MPS | MPSSum:
    """Nth-order derivative via Quantum Fourier Transform.

    Compute the `order`-th spatial derivative of a quantum state `psi`
    defined on a uniform grid of length `L`, using the QFT and momentum-space
    multiplication.

    Parameters
    ----------
    psi : MPS or MPSSum
        Quantum state in position representation
    L : float
        Length of the spatial domain
    order : int
        Order of the derivative

    Returns
    -------
    d_psi : MPS or MPSSum
        Quantum state encoding the derivative
    """
    N = psi.size
    dx = L / (2**N)
    return fourier_derivative_mpo(N, order, dx).apply(psi)
