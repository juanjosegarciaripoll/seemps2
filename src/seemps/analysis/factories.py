from __future__ import annotations
import numpy as np
from ..state import MPS
from .mesh import Interval, RegularInterval, ChebyshevInterval


def mps_equispaced(start: float, stop: float, sites: int) -> MPS:
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
    tensor_2 = np.zeros((2, 2, 1))
    tensor_2[:, :, 0] = np.array([[0, step], [1, 1]])
    tensors_bulk = [np.zeros((2, 2, 2)) for _ in range(sites - 2)]
    for idx, tensor in enumerate(tensors_bulk):
        tensor[0, :, 0] = np.ones(2)
        tensor[1, :, 1] = np.ones(2)
        tensor[0, 1, 1] = step * 2 ** (sites - (idx + 2))
    tensors = [tensor_1] + tensors_bulk + [tensor_2]
    return MPS(tensors)


def mps_exponential(start: float, stop: float, sites: int, c: complex = 1) -> MPS:
    """
    Returns an MPS representing an exponential function discretized over a
    half-open interval [start, stop).

    Parameters
    ----------
    start : float
        The start of the interval.
    stop : float
        The end of the interval.
    sites : int
        The number of sites or qubits for the MPS.
    c : complex, default=1
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
    for idx, tensor in enumerate(tensors_bulk):
        tensor[0, 0, 0] = 1
        tensor[0, 1, 0] = np.exp(c * step * 2 ** (sites - (idx + 2)))
    tensors = [tensor_1] + tensors_bulk + [tensor_2]
    return MPS(tensors)


def mps_sin(start: float, stop: float, sites: int) -> MPS:
    """
    Returns an MPS representing a sine function discretized over a
    half-open interval [start, stop).

    Parameters
    ----------
    start : float
        The start of the interval.
    stop : float
        The end of the interval.
    sites : int
        The number of sites or qubits for the MPS.
    strategy : Strategy, default=DEFAULT_STRATEGY
        The MPS simplification strategy to apply.

    Returns
    -------
    MPS
        An MPS representing the discretized sine function over the interval.
    """
    mps_1 = mps_exponential(start, stop, sites, c=1j)
    mps_2 = mps_exponential(start, stop, sites, c=-1j)
    return -0.5j * (mps_1 - mps_2).join()


def mps_cos(start: float, stop: float, sites: int) -> MPS:
    """
    Returns an MPS representing a cosine function discretized over a
    half-open interval [start, stop).

    Parameters
    ----------
    start : float
        The start of the interval.
    stop : float
        The end of the interval.
    sites : int
        The number of sites or qubits for the MPS.
    strategy : Strategy, default=DEFAULT_STRATEGY
        The MPS simplification strategy to apply.

    Returns
    -------
    MPS
        An MPS representing the discretized cosine function over the interval.
    """
    mps_1 = mps_exponential(start, stop, sites, c=1j)
    mps_2 = mps_exponential(start, stop, sites, c=-1j)
    return 0.5 * (mps_1 + mps_2).join()


def mps_affine(mps: MPS, orig: tuple[float, float], dest: tuple[float, float]) -> MPS:
    """
    Applies an affine transformation to an MPS, mapping it from one interval [x0, x1] to another [u0, u1].
    This is a transformation u = a * x + b, with u0 = a * x0 + b and and  u1 = a * x1 + b.
    Hence, a = (u1 - u0) / (x1 - x0) and b = ((u1 + u0) - a * (x0 + x1)) / 2.

    Parameters
    ----------
    mps : MPS | MPSSum
        The MPS to be transformed.
    orig : tuple[float, float]
        A tuple (x0, x1) representing the original interval.
    dest : tuple[float, float]
        A tuple (u0, u1) representing the destination interval.

    Returns
    -------
    mps_affine : MPS | MPSSum
        The MPS affinely transformed from (x0, x1) to (u0, u1).
    """
    x0, x1 = orig
    u0, u1 = dest
    a = (u1 - u0) / (x1 - x0)
    b = 0.5 * ((u1 + u0) - a * (x0 + x1))
    new_mps = a * mps
    if abs(b) > np.finfo(np.float64).eps:
        I = MPS([np.ones((1, 2, 1))] * new_mps.size)
        displaced_mps = new_mps + b * I
        return displaced_mps.join()
    return new_mps


def mps_interval(interval: Interval):
    """
    Returns an MPS corresponding to a specific type of interval.

    Parameters
    ----------
    interval : Interval
        The interval object containing start and stop points and the interval type.
        Currently supports `RegularInterval` and `ChebyshevInterval`.
    strategy : Strategy, default=DEFAULT_STRATEGY
        The MPS simplification strategy to apply.

    Returns
    -------
    MPS
        An MPS representing the interval according to its type.
    """
    start = interval.start
    stop = interval.stop
    sites = int(np.log2(interval.size))
    if isinstance(interval, RegularInterval):
        start_reg = start + interval.step if not interval.endpoint_left else start
        stop_reg = stop + interval.step if interval.endpoint_right else stop
        return mps_equispaced(start_reg, stop_reg, sites)
    elif isinstance(interval, ChebyshevInterval):
        if interval.endpoints is True:  # Extrema
            start_cheb = 0
            stop_cheb = np.pi + np.pi / (2**sites - 1)
        else:  # Zeros
            start_cheb = np.pi / (2 ** (sites + 1))
            stop_cheb = np.pi + start_cheb
        return mps_affine(
            mps_cos(start_cheb, stop_cheb, sites),
            (1, -1),  # Reverse order
            (start, stop),
        )
    else:
        raise ValueError(f"Unsupported interval type {type(interval)}")


__all__ = [
    "mps_equispaced",
    "mps_exponential",
    "mps_sin",
    "mps_cos",
    "mps_affine",
    "mps_interval",
]
