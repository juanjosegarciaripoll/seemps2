from __future__ import annotations
import numpy as np
from ...state import MPS, Strategy, DEFAULT_STRATEGY
from ...qft import iqft, qft_flip
from ..cross import (
    cross_interpolation,
    CrossStrategy,
    CrossStrategyMaxvol,
    BlackBoxLoadMPS,
)
from ..factories import mps_affine
from ..mesh import (
    IntegerInterval,
    Mesh,
    mps_to_mesh_matrix,
)


def mps_trapezoidal(start: float, stop: float, sites: int) -> MPS:
    """
    Returns the binary MPS representation of the trapezoidal quadrature on an interval.

    Parameters
    ----------
    start : float
        The starting point of the interval.
    stop : float
        The ending point of the interval.
    sites : int
        The number of sites or qubits for the MPS.
    """
    step = (stop - start) / (2**sites - 1)

    tensor_L = np.zeros((1, 2, 3))
    tensor_L[0, 0, 0] = 1
    tensor_L[0, 1, 1] = 1
    tensor_L[0, 0, 2] = 1
    tensor_L[0, 1, 2] = 1

    tensor_C = np.zeros((3, 2, 3))
    tensor_C[0, 0, 0] = 1
    tensor_C[1, 1, 1] = 1
    tensor_C[2, 0, 2] = 1
    tensor_C[2, 1, 2] = 1

    tensor_R = np.zeros((3, 2, 1))
    tensor_R[0, 0, 0] = -0.5
    tensor_R[1, 1, 0] = -0.5
    tensor_R[2, 0, 0] = 1
    tensor_R[2, 1, 0] = 1

    tensors = [tensor_L] + [tensor_C for _ in range(sites - 2)] + [tensor_R]
    return step * MPS(tensors)


def mps_simpson38(start: float, stop: float, sites: int) -> MPS:
    """
    Returns the binary MPS representation of the Simpson quadrature on an interval.
    Note that the number of sites must be even for Simpson's rule.

    Parameters
    ----------
    start : float
        The starting point of the interval.
    stop : float
        The ending point of the interval.
    sites : int
        The number of sites or qubits for the MPS. Must be even.
    """
    if sites % 2 != 0:
        raise ValueError("The number of sites must be even.")

    step = (stop - start) / (2**sites - 1)

    tensor_L1 = np.zeros((1, 2, 4))
    tensor_L1[0, 0, 0] = 1
    tensor_L1[0, 1, 1] = 1
    tensor_L1[0, 0, 2] = 1
    tensor_L1[0, 1, 3] = 1

    if sites == 2:
        tensor_R = np.zeros((4, 2, 1))
        tensor_R[0, 0, 0] = -1
        tensor_R[1, 1, 0] = -1
        tensor_R[2, 0, 0] = 2
        tensor_R[2, 1, 0] = 3
        tensor_R[3, 0, 0] = 3
        tensor_R[3, 1, 0] = 2
        tensors = [tensor_L1, tensor_R]
    else:
        tensor_L2 = np.zeros((4, 2, 5))
        tensor_L2[0, 0, 0] = 1
        tensor_L2[1, 1, 1] = 1
        tensor_L2[2, 0, 2] = 1
        tensor_L2[2, 1, 3] = 1
        tensor_L2[3, 0, 4] = 1
        tensor_L2[3, 1, 2] = 1

        tensor_C = np.zeros((5, 2, 5))
        tensor_C[0, 0, 0] = 1
        tensor_C[1, 1, 1] = 1
        tensor_C[2, 0, 2] = 1
        tensor_C[2, 1, 3] = 1
        tensor_C[3, 0, 4] = 1
        tensor_C[3, 1, 2] = 1
        tensor_C[4, 0, 3] = 1
        tensor_C[4, 1, 4] = 1

        tensor_R = np.zeros((5, 2, 1))
        tensor_R[0, 0, 0] = -1
        tensor_R[1, 1, 0] = -1
        tensor_R[2, 0, 0] = 2
        tensor_R[2, 1, 0] = 3
        tensor_R[3, 0, 0] = 3
        tensor_R[3, 1, 0] = 2
        tensor_R[4, 0, 0] = 3
        tensor_R[4, 1, 0] = 3

        tensors = (
            [tensor_L1, tensor_L2] + [tensor_C for _ in range(sites - 3)] + [tensor_R]
        )

    return (3 * step / 8) * MPS(tensors)


def mps_fifth_order(start: float, stop: float, sites: int) -> MPS:
    """
    Returns the binary MPS representation of the fifth-order quadrature on an interval.
    Note that the number of sites must be divisible by 4 for this quadrature rule.

    Parameters
    ----------
    start : float
        The starting point of the interval.
    stop : float
        The ending point of the interval.
    sites : int
        The number of sites or qubits for the MPS. Must be divisible by 4.
    """
    if sites % 4 != 0:
        raise ValueError("The number of sites must be divisible by 4.")

    step = (stop - start) / (2**sites - 1)

    tensor_L1 = np.zeros((1, 2, 4))
    tensor_L1[0, 0, 0] = 1
    tensor_L1[0, 1, 1] = 1
    tensor_L1[0, 0, 2] = 1
    tensor_L1[0, 1, 3] = 1

    tensor_L2 = np.zeros((4, 2, 6))
    tensor_L2[0, 0, 0] = 1
    tensor_L2[1, 1, 1] = 1
    tensor_L2[2, 0, 2] = 1
    tensor_L2[2, 1, 3] = 1
    tensor_L2[3, 0, 4] = 1
    tensor_L2[3, 1, 5] = 1

    tensor_L3 = np.zeros((6, 2, 7))
    tensor_L3[0, 0, 0] = 1
    tensor_L3[1, 1, 1] = 1
    tensor_L3[2, 0, 2] = 1
    tensor_L3[2, 1, 3] = 1
    tensor_L3[3, 0, 4] = 1
    tensor_L3[3, 1, 5] = 1
    tensor_L3[4, 0, 6] = 1
    tensor_L3[4, 1, 2] = 1
    tensor_L3[5, 0, 3] = 1
    tensor_L3[5, 1, 4] = 1

    tensor_C = np.zeros((7, 2, 7))
    tensor_C[0, 0, 0] = 1
    tensor_C[1, 1, 1] = 1
    tensor_C[2, 0, 2] = 1
    tensor_C[2, 1, 3] = 1
    tensor_C[3, 0, 4] = 1
    tensor_C[3, 1, 5] = 1
    tensor_C[4, 0, 6] = 1
    tensor_C[4, 1, 2] = 1
    tensor_C[5, 0, 3] = 1
    tensor_C[5, 1, 4] = 1
    tensor_C[6, 0, 5] = 1
    tensor_C[6, 1, 6] = 1

    tensor_R = np.zeros((7, 2, 1))
    tensor_R[0, 0, 0] = -19
    tensor_R[1, 1, 0] = -19
    tensor_R[2, 0, 0] = 38
    tensor_R[2, 1, 0] = 75
    tensor_R[3, 0, 0] = 50
    tensor_R[3, 1, 0] = 50
    tensor_R[4, 0, 0] = 75
    tensor_R[4, 1, 0] = 38
    tensor_R[5, 0, 0] = 75
    tensor_R[5, 1, 0] = 50
    tensor_R[6, 0, 0] = 50
    tensor_R[6, 1, 0] = 75

    tensors = (
        [tensor_L1, tensor_L2, tensor_L3]
        + [tensor_C for _ in range(sites - 4)]
        + [tensor_R]
    )
    return (5 * step / 288) * MPS(tensors)


def mps_best_newton_cotes(start: float, stop: float, sites: int) -> MPS:
    """Fetches the MPS for the best Newton-Côtes quadrature rule for the given sites."""
    if sites % 4 == 0:
        return mps_fifth_order(start, stop, sites)
    elif sites % 2 == 0:
        return mps_simpson38(start, stop, sites)
    else:
        return mps_trapezoidal(start, stop, sites)


def mps_fejer(
    start: float,
    stop: float,
    sites: int,
    qft_strategy: Strategy = DEFAULT_STRATEGY,
    cross_strategy: CrossStrategy = CrossStrategyMaxvol(),
) -> MPS:
    """
    Returns the binary MPS representation of the Fejér first quadrature rule on an interval.
    The integration nodes are given by the `d` zeros of the `d`-th Chebyshev polynomial.
    This is achieved using the formulation of Waldvogel (see waldvogel2006 formula 4.4)
    by means of a direct encoding of the Féjer phase, tensor-cross interpolation
    for the term $1/(1-4*k**2)$, and the bit-flipped inverse Quantum Fourier Transform (iQFT).

    Parameters
    ----------
    start : float
        The start of the interval.
    stop : float
        The end of the interval.
    sites : int
        The number of sites or qubits for the MPS.
    strategy : Strategy, default=DEFAULT_STRATEGY
        The strategy for MPS simplification.
    cross_strategy : CrossStrategyDMRG, default=CrossStrategyDMRG.
        The strategy for tensor cross-interpolation.
    """
    N = int(2**sites)

    # Encode 1/(1 - 4*k**2) term with TCI
    def selector(k: np.ndarray) -> np.ndarray:
        return np.where(k < N / 2, 2 / (1 - 4 * k**2), 2 / (1 - 4 * (N - k) ** 2))

    black_box = BlackBoxLoadMPS(
        selector,
        mesh=Mesh([IntegerInterval(0, N)]),
        map_matrix=mps_to_mesh_matrix([sites]),
        physical_dimensions=[2] * sites,
    )
    mps_k2 = cross_interpolation(black_box, cross_strategy).mps

    # Encode phase term analytically
    p = 1j * np.pi / N  # prefactor
    exponent = p * 2 ** (sites - 1)

    tensor_L = np.zeros((1, 2, 5), dtype=complex)
    tensor_L[0, 0, 0] = 1
    tensor_L[0, 1, 1] = np.exp(-exponent)
    tensor_L[0, 1, 2] = np.exp(exponent)
    tensor_L[0, 1, 3] = -np.exp(-exponent)
    tensor_L[0, 1, 4] = -np.exp(exponent)

    tensor_R = np.zeros((5, 2, 1), dtype=complex)
    tensor_R[0, 0, 0] = 1
    tensor_R[0, 1, 0] = np.exp(p)
    tensor_R[1, 0, 0] = 1
    tensor_R[1, 1, 0] = np.exp(p)
    tensor_R[2, 0, 0] = 1
    tensor_R[3, 0, 0] = 1
    tensor_R[4, 0, 0] = 1

    tensors_C = [np.zeros((5, 2, 5), dtype=complex) for _ in range(sites - 2)]
    for idx, tensor_C in enumerate(tensors_C):
        expn = p * 2 ** (sites - (idx + 2))
        tensor_C[0, 0, 0] = 1
        tensor_C[0, 1, 0] = np.exp(expn)
        tensor_C[1, 0, 1] = 1
        tensor_C[1, 1, 1] = np.exp(expn)
        tensor_C[2, 0, 2] = 1
        tensor_C[3, 0, 3] = 1
        tensor_C[4, 0, 4] = 1

    tensors = [tensor_L] + tensors_C + [tensor_R]
    mps_phase = MPS(tensors)

    # Encode Fejér quadrature with iQFT
    mps = (1 / np.sqrt(2) ** sites) * qft_flip(
        iqft(mps_k2 * mps_phase, strategy=qft_strategy)
    )

    return mps_affine(mps, (-1, 1), (start, stop))


def mps_clenshaw_curtis(
    start: float,
    stop: float,
    sites: int,
    strategy: Strategy = DEFAULT_STRATEGY,
) -> MPS:
    """
    Returns the binary MPS representation of the Clenshaw-Curtis quadrature rule on an interval.
    The integration nodes are given by the `d+1` extrema of the `d`-th Chebyshev polynomial.
    This is achieved using the formulation of Waldvogel (see waldvogel2006 formula 4.2) using
    the Schmidt decomposition.

    Parameters
    ----------
    start : float
        The start of the interval.
    stop : float
        The end of the interval.
    sites : int
        The number of sites or qubits for the MPS.
    strategy : Strategy, default=DEFAULT_STRATEGY.
        The strategy for the Schmidt decomposition.
    """
    # TODO: Find a way to construct the MPS analytically without using SVD.
    # Problem: it cannot be directly computed as the iFFT of a vector of size 2**n
    # thus, it cannot be constructed as the iQFT of another MPS.
    N = int(2**sites) - 1

    # Construct the quadrature vector using the iFFT
    v = np.zeros(N)
    g = np.zeros(N)
    w0 = 1 / (N**2 - 1 + (N % 2))
    for k in range(N // 2):
        v[k] = 2 / (1 - 4 * k**2)
        g[k] = -w0
    v[N // 2] = (N - 3) / (2 * (N // 2) - 1) - 1
    g[N // 2] = w0 * ((2 - (N % 2)) * N - 1)
    for k in range(1, N // 2 + 1):
        v[-k] = v[k]
        g[-k] = g[k]
    w = np.fft.ifft(v + g).real
    w = np.hstack((w, w[0]))

    # Decompose the quadrature vector with the Schmidt decomposition
    mps = MPS.from_vector(w, [2] * sites, strategy=strategy, normalize=False)
    return mps_affine(mps, (-1, 1), (start, stop))  # type: ignore
