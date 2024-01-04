import numpy as np
from typing import List

from ..state import MPS, Strategy, DEFAULT_STRATEGY
from ..truncate import simplify
from .mesh import (
    Interval,
    RegularClosedInterval,
    RegularHalfOpenInterval,
    ChebyshevZerosInterval,
)


def mps_equispaced(start: float, stop: float, sites: int):
    """
    Returns an MPS representing a discretized interval with equispaced points.

    Parameters
    ----------
    start : float
        The start of the interval.
    stop : float
        The end of the interval.
    sites : int
        The number of sites or qubits for the MPS.

    Returns
    -------
    MPS
        An MPS representing an equispaced discretization within [start, stop].
    """
    step = (stop - start) / 2**sites
    tensor_1 = np.zeros((1, 2, 2))
    tensor_1[0, :, :] = np.array([[[1, start], [1, start + step * 2 ** (sites - 1)]]])
    tensors_bulk = [np.zeros((2, 2, 2)) for _ in range(sites - 2)]
    for i in range(len(tensors_bulk)):
        tensors_bulk[i][0, :, 0] = np.ones(2)
        tensors_bulk[i][1, :, 1] = np.ones(2)
        tensors_bulk[i][0, 1, 1] = step * 2 ** (sites - (i + 2))
    tensor_2 = np.zeros((2, 2, 1))
    tensor_2[:, :, 0] = np.array([[0, step], [1, 1]])
    tensors = [tensor_1] + tensors_bulk + [tensor_2]
    return MPS(tensors)


def mps_exponential(start: float, stop: float, sites: int, c: complex = 1) -> MPS:
    """
    Returns an MPS representing an exponential function discretized over an interval.

    Parameters
    ----------
    start : float
        The start of the interval.
    stop : float
        The end of the interval.
    sites : int
        The number of sites or qubits for the MPS.
    c : complex
        The coefficient in the exponent of the exponential function.

    Returns
    -------
    MPS
        An MPS representing the discretized exponential function over the interval.
    """
    step = (stop - start) / 2**sites
    tensor_1 = np.zeros((1, 2, 1), dtype=complex)
    tensor_1[0, 0, 0] = np.exp(c * start)
    tensor_1[0, 1, 0] = np.exp(c * start + c * step * 2 ** (sites - 1))

    tensor_2 = np.zeros((1, 2, 1), dtype=complex)
    tensor_2[0, 0, 0] = 1
    tensor_2[0, 1, 0] = np.exp(c * step)

    tensors_bulk = [np.zeros((1, 2, 1), dtype=complex) for _ in range(sites - 2)]
    for i in range(len(tensors_bulk)):
        tensors_bulk[i][0, 0, 0] = 1
        tensors_bulk[i][0, 1, 0] = np.exp(c * step * 2 ** (sites - (i + 2)))
    tensors = [tensor_1] + tensors_bulk + [tensor_2]
    return MPS(tensors)


def mps_sine(
    start: float, stop: float, sites: int, strategy: Strategy = DEFAULT_STRATEGY
) -> MPS:
    """
    Returns an MPS representing a sine function discretized over an interval.

    Parameters
    ----------
    start : float
        The start of the interval.
    stop : float
        The end of the interval.
    sites : int
        The number of sites or qubits for the MPS.

    Returns
    -------
    MPS
        An MPS representing the discretized sine function over the interval.
    """
    mps_1 = mps_exponential(start, stop, sites, c=1j)
    mps_2 = mps_exponential(start, stop, sites, c=-1j)

    return -0.5j * simplify(mps_1 - mps_2, strategy=strategy)


def mps_cosine(
    start: float, stop: float, sites: int, strategy: Strategy = DEFAULT_STRATEGY
) -> MPS:
    """
    Returns an MPS representing a cosine function discretized over an interval.

    Parameters
    ----------
    start : float
        The start of the interval.
    stop : float
        The end of the interval.
    sites : int
        The number of sites or qubits for the MPS.

    Returns
    -------
    MPS
        An MPS representing the discretized cosine function over the interval.
    """
    mps_1 = mps_exponential(start, stop, sites, c=1j)
    mps_2 = mps_exponential(start, stop, sites, c=-1j)

    return 0.5 * simplify(mps_1 + mps_2, strategy=strategy)


def mps_interval(interval: Interval, strategy: Strategy = DEFAULT_STRATEGY):
    """
    Returns an MPS corresponding to a specific type of interval (open, closed, or Chebyshev zeros).

    Parameters
    ----------
    interval : Interval
        The interval object containing start and stop points and the interval type.

    Returns
    -------
    MPS
        An MPS representing the interval according to its type.
    """
    start = interval.start
    stop = interval.stop
    sites = int(np.log2(interval.size))
    if isinstance(interval, RegularHalfOpenInterval):
        return mps_equispaced(start, stop, sites)
    elif isinstance(interval, RegularClosedInterval):
        stop += (stop - start) / (2**sites - 1)
        return mps_equispaced(start, stop, sites)
    elif isinstance(interval, ChebyshevZerosInterval):
        start_mapped = np.pi / (2 ** (sites + 1))
        stop_mapped = np.pi + start_mapped
        return -1.0 * mps_cosine(start_mapped, stop_mapped, sites, strategy=strategy)
    else:
        raise ValueError(f"Unsupported interval type {type(interval)}")


def mps_tensor_product(mps_list: List[MPS]) -> MPS:
    """
    Returns the tensor product of a list of MPS.

    Parameters
    ----------
    mps_list : List[MPS]
        The list of MPS objects to multiply.

    Returns
    -------
    MPS
        The resulting MPS from the tensor product of the input list.
    """
    nested_sites = [mps._data for mps in mps_list]
    flattened_sites = [site for sites in nested_sites for site in sites]
    return MPS(flattened_sites)


def mps_tensor_sum(mps_list: List[MPS], strategy: Strategy = DEFAULT_STRATEGY) -> MPS:
    """
    Returns the tensor sum of a list of MPS.

    Parameters
    ----------
    mps_list : List[MPS]
        The list of MPS objects to sum.
    strategy : Strategy, optional
        The MPS simplification strategy to apply.

    Returns
    -------
    MPS
        The resulting MPS from the summation of the list.
    """
    total_sites = np.sum([len(mps) for mps in mps_list])
    result = MPS([np.zeros((1, 2, 1))] * total_sites)
    for idx, mps in enumerate(mps_list):
        term = [MPS([np.ones((1, 2, 1))] * len(mps)) for mps in mps_list]
        term[idx] = mps
        result = simplify(result + mps_tensor_product(term), strategy=strategy)
    return result
