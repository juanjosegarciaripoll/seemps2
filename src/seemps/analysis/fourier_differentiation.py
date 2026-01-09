import numpy as np
from seemps.state import MPS, MPSSum
from seemps.qft import qft, iqft, qft_flip
from seemps.analysis.operators import p_to_n_mpo, p_mpo

def fourier_derivative(psi: MPS | MPSSum, L: float, order: int) -> MPS | MPSSum:
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

    # QFT
    psi_k_reversed_mps = qft(psi)
    psi_k_mps = qft_flip(psi_k_reversed_mps)

    # Apply momentum operator
    if order == 1:
        P_op = p_mpo(N, dx)
    else:
        P_op = p_to_n_mpo(N, dx, order)
    d_psi_k_mps = (1j)**order * (P_op @ psi_k_mps)

    # Inverse QFT
    d_psi_x_reversed_mps = iqft(d_psi_k_mps)
    d_psi_x_mps = qft_flip(d_psi_x_reversed_mps)

    return d_psi_x_mps