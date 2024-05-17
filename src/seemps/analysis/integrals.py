from __future__ import annotations
import numpy as np
from typing import Callable, Union
from math import sqrt
from ..truncate import simplify
from ..state import MPS, Strategy, scprod
from ..qft import iqft, qft_flip
from .mesh import RegularInterval, Mesh
from .factories import mps_tensor_product, mps_affine, COMPUTER_PRECISION
from .cross import cross_maxvol, BlackBoxLoadMPS, CrossStrategyMaxvol


def mps_midpoint(start: float, stop: float, sites: int) -> MPS:
    """
    Returns a MPS representing the midpoint quadrature of an interval.

    Parameters
    ---------
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
    Returns a MPS representing the trapezoidal quadrature of an interval.

    Parameters
    ---------
    start : float
        The starting point of the interval.
    stop : float
        The ending point of the interval.
    sites : int
        The number of sites or qubits for the MPS.
    """
    tensor_1 = np.zeros((1, 2, 3))
    tensor_1[0, 0, 0] = 1
    tensor_1[0, 1, 1] = 1
    tensor_1[0, 0, 2] = 1
    tensor_1[0, 1, 2] = 1
    tensor_bulk = np.zeros((3, 2, 3))
    tensor_bulk[0, 0, 0] = 1
    tensor_bulk[1, 1, 1] = 1
    tensor_bulk[2, 0, 2] = 1
    tensor_bulk[2, 1, 2] = 1
    tensor_2 = np.zeros((3, 2, 1))
    tensor_2[0, 0, 0] = -0.5
    tensor_2[1, 1, 0] = -0.5
    tensor_2[2, 0, 0] = 1
    tensor_2[2, 1, 0] = 1
    tensors = [tensor_1] + [tensor_bulk for _ in range(sites - 2)] + [tensor_2]
    step = (stop - start) / (2**sites - 1)
    return step * MPS(tensors)


def mps_simpson(start: float, stop: float, sites: int) -> MPS:
    """
    Returns a MPS representing the Simpson quadrature of an interval.
    Note that the number of sites must be even for Simpson's rule.

    Parameters
    ---------
    start : float
        The starting point of the interval.
    stop : float
        The ending point of the interval.
    sites : int
        The number of sites or qubits for the MPS. Must be even.
    """
    if sites % 2 != 0:
        raise ValueError("The sites must be divisible by 2.")

    tensor_1 = np.zeros((1, 2, 4))
    tensor_1[0, 0, 0] = 1
    tensor_1[0, 1, 1] = 1
    tensor_1[0, 0, 2] = 1
    tensor_1[0, 1, 3] = 1
    if sites == 2:
        tensor_2 = np.zeros((4, 2, 1))
        tensor_2[0, 0, 0] = -1
        tensor_2[1, 1, 0] = -1
        tensor_2[2, 0, 0] = 2
        tensor_2[2, 1, 0] = 3
        tensor_2[3, 0, 0] = 3
        tensor_2[3, 1, 0] = 2
        tensors = [tensor_1, tensor_2]
    else:
        tensor_2 = np.zeros((4, 2, 5))
        tensor_2[0, 0, 0] = 1
        tensor_2[1, 1, 1] = 1
        tensor_2[2, 0, 2] = 1
        tensor_2[2, 1, 3] = 1
        tensor_2[3, 0, 4] = 1
        tensor_2[3, 1, 2] = 1
        tensor_bulk = np.zeros((5, 2, 5))
        tensor_bulk[0, 0, 0] = 1
        tensor_bulk[1, 1, 1] = 1
        tensor_bulk[2, 0, 2] = 1
        tensor_bulk[2, 1, 3] = 1
        tensor_bulk[3, 0, 4] = 1
        tensor_bulk[3, 1, 2] = 1
        tensor_bulk[4, 0, 3] = 1
        tensor_bulk[4, 1, 4] = 1
        tensor_3 = np.zeros((5, 2, 1))
        tensor_3[0, 0, 0] = -1
        tensor_3[1, 1, 0] = -1
        tensor_3[2, 0, 0] = 2
        tensor_3[2, 1, 0] = 3
        tensor_3[3, 0, 0] = 3
        tensor_3[3, 1, 0] = 2
        tensor_3[4, 0, 0] = 3
        tensor_3[4, 1, 0] = 3
        tensors = (
            [tensor_1, tensor_2] + [tensor_bulk for _ in range(sites - 3)] + [tensor_3]
        )
    step = (stop - start) / (2**sites - 1)
    return (3 * step / 8) * MPS(tensors)


def mps_fifth_order(start: float, stop: float, sites: int) -> MPS:
    """
    Returns a MPS representing the fifth-order quadrature of an interval.
    Note that the number of sites must be divisible by 4 for this quadrature rule.

    Parameters
    ---------
    start : float
        The starting point of the interval.
    stop : float
        The ending point of the interval.
    sites : int
        The number of sites or qubits for the MPS. Must be divisible by 4.
    """
    if sites % 4 != 0:
        raise ValueError("The sites must be divisible by 4.")
    tensor_1 = np.zeros((1, 2, 4))
    tensor_1[0, 0, 0] = 1
    tensor_1[0, 1, 1] = 1
    tensor_1[0, 0, 2] = 1
    tensor_1[0, 1, 3] = 1
    tensor_2 = np.zeros((4, 2, 6))
    tensor_2[0, 0, 0] = 1
    tensor_2[1, 1, 1] = 1
    tensor_2[2, 0, 2] = 1
    tensor_2[2, 1, 3] = 1
    tensor_2[3, 0, 4] = 1
    tensor_2[3, 1, 5] = 1
    tensor_3 = np.zeros((6, 2, 7))
    tensor_3[0, 0, 0] = 1
    tensor_3[1, 1, 1] = 1
    tensor_3[2, 0, 2] = 1
    tensor_3[2, 1, 3] = 1
    tensor_3[3, 0, 4] = 1
    tensor_3[3, 1, 5] = 1
    tensor_3[4, 0, 6] = 1
    tensor_3[4, 1, 2] = 1
    tensor_3[5, 0, 3] = 1
    tensor_3[5, 1, 4] = 1
    tensor_bulk = np.zeros((7, 2, 7))
    tensor_bulk[0, 0, 0] = 1
    tensor_bulk[1, 1, 1] = 1
    tensor_bulk[2, 0, 2] = 1
    tensor_bulk[2, 1, 3] = 1
    tensor_bulk[3, 0, 4] = 1
    tensor_bulk[3, 1, 5] = 1
    tensor_bulk[4, 0, 6] = 1
    tensor_bulk[4, 1, 2] = 1
    tensor_bulk[5, 0, 3] = 1
    tensor_bulk[5, 1, 4] = 1
    tensor_bulk[6, 0, 5] = 1
    tensor_bulk[6, 1, 6] = 1
    tensor_4 = np.zeros((7, 2, 1))
    tensor_4[0, 0, 0] = -19
    tensor_4[1, 1, 0] = -19
    tensor_4[2, 0, 0] = 38
    tensor_4[2, 1, 0] = 75
    tensor_4[3, 0, 0] = 50
    tensor_4[3, 1, 0] = 50
    tensor_4[4, 0, 0] = 75
    tensor_4[4, 1, 0] = 38
    tensor_4[5, 0, 0] = 75
    tensor_4[5, 1, 0] = 50
    tensor_4[6, 0, 0] = 50
    tensor_4[6, 1, 0] = 75
    tensors = (
        [tensor_1, tensor_2, tensor_3]
        + [tensor_bulk for _ in range(sites - 4)]
        + [tensor_4]
    )
    step = (stop - start) / (2**sites - 1)
    return (5 * step / 288) * MPS(tensors)


def mps_fejer(
    start: float,
    stop: float,
    sites: int,
    strategy: Strategy = COMPUTER_PRECISION,
    cross_strategy: CrossStrategyMaxvol = CrossStrategyMaxvol(tol_sampling=1e-15),
) -> MPS:
    """
    Returns the MPS encoding of the Fejér first quadrature rule.
    This is achieved using the formulation of Waldvogel (see waldvogel2006 formula 4.4)
    by means of a direct encoding of the Féjer phase, tensor cross interpolation
    for the term $1/(1-4*k**2)$, and the corrected inverse Quantum Fourier Transform (iQFT).

    Parameters
    ----------
    start : float
        The start of the interval.
    stop : float
        The end of the interval.
    sites : int
        The number of sites or qubits for the MPS.
    strategy : Optional[Strategy], optional
        The strategy for MPS simplification. Defaults to a tolerance of 1e-15.
    cross_strategy : CrossStrategy, optional
        The strategy for cross interpolation. Defaults to a tolerance of 1e-15.

    Returns
    -------
    MPS
        An MPS encoding of the Fejér first quadrature rule.
    """

    N = int(2**sites)

    # Encode 1/(1 - 4*k**2) term with cross interpolation
    def func(k):
        return np.where(
            k < N / 2,
            2 / (1 - 4 * k**2),
            2 / (1 - 4 * (N - k) ** 2),
        )

    cross_results = cross_maxvol(
        BlackBoxLoadMPS(func, Mesh([RegularInterval(0, N, N)])),
        cross_strategy=cross_strategy,
    )
    mps_k2 = simplify(cross_results.mps, strategy=strategy)

    # Encode phase term analytically
    p = 1j * np.pi / N  # prefactor
    exponent = p * 2 ** (sites - 1)
    tensor_1 = np.zeros((1, 2, 5), dtype=complex)
    tensor_2 = np.zeros((5, 2, 1), dtype=complex)
    tensors_bulk = [np.zeros((5, 2, 5), dtype=complex) for _ in range(sites - 2)]
    tensor_1[0, 0, 0] = 1
    tensor_1[0, 1, 1] = np.exp(-exponent)
    tensor_1[0, 1, 2] = np.exp(exponent)
    tensor_1[0, 1, 3] = -np.exp(-exponent)
    tensor_1[0, 1, 4] = -np.exp(exponent)
    tensor_2[0, 0, 0] = 1
    tensor_2[0, 1, 0] = np.exp(p)
    tensor_2[1, 0, 0] = 1
    tensor_2[1, 1, 0] = np.exp(p)
    tensor_2[2, 0, 0] = 1
    tensor_2[3, 0, 0] = 1
    tensor_2[4, 0, 0] = 1
    for idx, tensor in enumerate(tensors_bulk):
        exponent = p * 2 ** (sites - (idx + 2))
        tensor[0, 0, 0] = 1
        tensor[0, 1, 0] = np.exp(exponent)
        tensor[1, 0, 1] = 1
        tensor[1, 1, 1] = np.exp(exponent)
        tensor[2, 0, 2] = 1
        tensor[3, 0, 3] = 1
        tensor[4, 0, 4] = 1
    tensors = [tensor_1] + tensors_bulk + [tensor_2]
    mps_phase = MPS(tensors)

    # Encode Fejér MPS with iQFT
    mps_v = mps_k2 * mps_phase
    mps = (1 / sqrt(2) ** sites) * qft_flip(iqft(mps_v, strategy=strategy))

    return mps_affine(mps, (-1, 1), (start, stop)).as_mps()


def integrate_mps(
    mps: MPS, mesh: Mesh, integral_type: str = "trapezoidal", mps_order: str = "A"
) -> Union[float, complex]:
    """
    Returns the integral of a MPS representation of a function defined on a given mesh.
    Supports multiple quadrature types, including midpoint, trapezoidal, Simpson's rule,
    fifth-order, and Fejér's rule.

    Parameters
    ----------
    mps : MPS
        The input MPS to integrate representing a multivariate function.
    mesh : Mesh
        A Mesh object representing the intervals over which the function is defined.
    integral_type : str
        The type of numerical integration method to use. Options:
        - "midpoint": Midpoint rule integration.
        - "trapezoidal": Trapezoidal rule integration (default).
        - "simpson": Simpson's rule integration (requires the qubits of each MPS to be a multiple of 2).
        - "fifth_order": Fifth-order rule integration (requires the qubits of each MPS to be a multiple of 4).
        - "fejer": Fejér's rule integration (requires the function to be discretized on Chebyshev zeros).
        If an unsupported integral type is specified, a ValueError is raised.
    mps_order : str, optional
        The order in which to arrange the qubits of the quadrature before contraction. Options:
        - 'A': The qubits are arranged by dimension (default).
        - 'B': The qubits are arranged by significance.

    Returns
    -------
    Union[float, complex]
        The resulting integral.
    """
    foo: Callable[[float, float, int], MPS]
    if integral_type == "midpoint":
        foo = mps_midpoint
    elif integral_type == "trapezoidal":
        foo = mps_trapezoidal
    elif integral_type == "simpson" and len(mps) % 2 == 0:
        foo = mps_simpson
    elif integral_type == "fifth_order" and len(mps) % 4 == 0:
        foo = mps_fifth_order
    elif integral_type == "fejer":
        foo = mps_fejer
    else:
        raise ValueError("Invalid integral_type or number of sites")

    mps_list = []
    for interval in mesh.intervals:
        mps_list.append(foo(interval.start, interval.stop, int(np.log2(interval.size))))
    return scprod(mps, mps_tensor_product(mps_list, mps_order=mps_order))
