import numpy as np
from scipy.fftpack import ifft  # type: ignore
from typing import Callable, Union
from seemps.analysis import Mesh, mps_tensor_product
from seemps.expectation import scprod
from seemps.state import DEFAULT_TOLERANCE, MPS, Strategy

QUADRATURE_STRATEGY = Strategy(tolerance=DEFAULT_TOLERANCE)


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


def mps_fejer(start: float, stop: float, points: int) -> MPS:
    """
    Returns a MPS representing the Féjer quadrature of an interval.

    Parameters
    ---------
    start : float
        The starting point of the interval.
    stop : float
        The ending point of the interval.
    points : int
        The number of quadrature points.
    """
    # TODO: Optimize maybe this?
    d = 2**points
    N = np.arange(start=1, stop=d, step=2)[:, None]
    l = N.size
    v0 = [2 * np.exp(1j * np.pi * k / d) / (1 - 4 * k**2) for k in range(l)] + [0] * (
        l + 1
    )
    v1 = v0[0:-1] + np.conj(v0[:0:-1])
    vector = ifft(v1).flatten().real
    mps = MPS.from_vector(
        vector,
        [2 for _ in range(points)],
        normalize=False,
        strategy=QUADRATURE_STRATEGY,
    )
    step = (stop - start) / 2
    return step * mps


def integrate_mps(mps: MPS, mesh: Mesh, integral_type: str) -> Union[float, complex]:
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
        raise ValueError("Invalid integral_type")

    mps_list = []
    for interval in mesh.intervals:
        mps_list.append(foo(interval.start, interval.stop, int(np.log2(interval.size))))
    return scprod(mps, mps_tensor_product(mps_list))
