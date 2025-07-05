from __future__ import annotations
import numpy as np
from math import sqrt
from ..state import MPS, Strategy, scprod, DEFAULT_STRATEGY
from ..truncate import simplify
from ..qft import iqft, qft_flip
from .mesh import Mesh, Interval, RegularInterval, ChebyshevInterval, IntegerInterval
from .factories import mps_affine, mps_tensor_product
from .cross import cross_dmrg, BlackBoxLoadMPS, CrossStrategyDMRG

# TODO: Express the quadratures in terms of a 'nodes' and 'quantize' arguments, and
# implement 'mps_trapezoidal' for any base.


def integrate_mps(mps: MPS, domain: Interval | Mesh, mps_order: str = "A") -> complex:
    """
    Returns the integral of a multivariate function represented as a MPS.
    Uses the 'best possible' quadrature rule according to the intervals that compose the mesh.
    Intervals of type `RegularInterval` employ high-order Newton-Côtes rules, while
    those of type `ChebyshevInterval` employ Clenshaw-Curtis rules.

    Parameters
    ----------
    mps : MPS
        The MPS representation of the multivariate function to be integrated.
    domain : Interval | Mesh
        An object defining the discretization domain of the function.
        Can be either an `Interval` or a `Mesh` given by a collection of intervals.
        The quadrature rules are selected based on the properties of these intervals.
    mps_order : str, default='A'
        Specifies the ordering of the qubits in the quadrature. Possible values:.
        - 'A': Qubits are serially ordered (by variable).
        - 'B': Qubits are interleaved (by significance).

    Returns
    -------
    complex
        The integral of the MPS representation of the function discretized in the given Mesh.

    Notes
    -----
    - This algorithm assumes that all function variables are in the standard MPS form, i.e.
    quantized in base 2, and are discretized either on a `RegularInterval or `ChebyshevInterval`.

    - For more general structures, the quadrature MPS can be constructed
    using the univariate quadrature rules and the `mps_tensor_product` routine, which can be
    subsequently contracted using the `scprod` routine.

    Examples
    --------
    .. code-block:: python

        # Integrate a given bivariate function using the Clenshaw-Curtis quadrature.
        # Assuming that the MPS is already loaded (for example, using TT-Cross or Chebyshev).
        mps_function_2d = ...

        # Define a domain that matches the MPS to integrate.
        start, stop = -1, 1
        n_qubits = 10
        interval = ChebyshevInterval(-1, 1, 2**n_qubits, endpoints=True)
        mesh = Mesh([interval, interval])

        # Integrate the MPS on the given discretization domain.
        integral = integrate_mps(mps_function_2d, mesh)
    """
    mesh = domain if isinstance(domain, Mesh) else Mesh([domain])
    quads = []
    for interval in mesh.intervals:
        a, b, N = interval.start, interval.stop, interval.size
        n = int(np.log2(N))
        if isinstance(interval, RegularInterval):
            if n % 4 == 0:
                quads.append(mps_fifth_order(a, b, n))
            elif n % 2 == 0:
                quads.append(mps_simpson(a, b, n))
            else:
                quads.append(mps_trapezoidal(a, b, n))
        elif isinstance(interval, ChebyshevInterval):
            if interval.endpoints:
                quads.append(mps_clenshaw_curtis(a, b, n))
            else:
                quads.append(mps_fejer(a, b, n))
        else:
            raise ValueError("Invalid interval in mesh")
    mps_quad = quads[0] if len(quads) == 1 else mps_tensor_product(quads, mps_order)
    return scprod(mps, mps_quad)


# TODO: Consider removing this (trapezoidal is strictly superior)
def mps_midpoint(start: float, stop: float, sites: int) -> MPS:
    """
    Returns the binary MPS representation of the midpoint quadrature on an interval.

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
    return step * MPS([np.ones((1, 2, 1))] * sites)


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
    tensor_L = np.zeros((1, 2, 3))  # Left
    tensor_L[0, 0, 0] = 1
    tensor_L[0, 1, 1] = 1
    tensor_L[0, 0, 2] = 1
    tensor_L[0, 1, 2] = 1
    tensor_C = np.zeros((3, 2, 3))  # Center
    tensor_C[0, 0, 0] = 1
    tensor_C[1, 1, 1] = 1
    tensor_C[2, 0, 2] = 1
    tensor_C[2, 1, 2] = 1
    tensor_R = np.zeros((3, 2, 1))  # Right
    tensor_R[0, 0, 0] = -0.5
    tensor_R[1, 1, 0] = -0.5
    tensor_R[2, 0, 0] = 1
    tensor_R[2, 1, 0] = 1
    tensors = [tensor_L] + [tensor_C for _ in range(sites - 2)] + [tensor_R]
    step = (stop - start) / (2**sites - 1)
    return step * MPS(tensors)


def mps_simpson(start: float, stop: float, sites: int) -> MPS:
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
        raise ValueError("The sites must be divisible by 2.")

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
    step = (stop - start) / (2**sites - 1)
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
        raise ValueError("The sites must be divisible by 4.")
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
    step = (stop - start) / (2**sites - 1)
    return (5 * step / 288) * MPS(tensors)


def mps_fejer(
    start: float,
    stop: float,
    sites: int,
    strategy: Strategy = DEFAULT_STRATEGY,
    cross_strategy: CrossStrategyDMRG = CrossStrategyDMRG(),
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

    mps_k2 = cross_dmrg(
        BlackBoxLoadMPS(selector, IntegerInterval(0, N)), cross_strategy=cross_strategy
    ).mps
    mps_k2 = simplify(mps_k2, strategy=strategy)

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
        exponent = p * 2 ** (sites - (idx + 2))
        tensor_C[0, 0, 0] = 1
        tensor_C[0, 1, 0] = np.exp(exponent)
        tensor_C[1, 0, 1] = 1
        tensor_C[1, 1, 1] = np.exp(exponent)
        tensor_C[2, 0, 2] = 1
        tensor_C[3, 0, 3] = 1
        tensor_C[4, 0, 4] = 1
    tensors = [tensor_L] + tensors_C + [tensor_R]
    mps_phase = MPS(tensors)

    # Encode Fejér quadrature with iQFT
    mps = (1 / sqrt(2) ** sites) * qft_flip(iqft(mps_k2 * mps_phase, strategy=strategy))

    return mps_affine(mps, (-1, 1), (start, stop))  # type: ignore


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
