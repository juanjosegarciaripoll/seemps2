from __future__ import annotations
import math
from typing import Sequence
import numpy as np
from ..state import MPS
from .mesh import (
    Interval,
    RegularInterval,
    ChebyshevInterval,
    QuantizedInterval,
    IntervalTuple,
)


def _mps_equispaced(interval: QuantizedInterval | IntervalTuple) -> MPS:
    """
    Returns an MPS representing a discretized interval with equispaced points.

    Parameters
    ----------
    interval: QuantizedInterval | IntervalTuple
        The interval over which the function is defined.

    Returns
    -------
    MPS
        An MPS representing an equispaced discretization within [start, stop].
    """
    if isinstance(interval, tuple):
        interval = QuantizedInterval(*interval)
    start, sites, step = (
        interval[0],
        interval.qubits,
        interval.step,
    )
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


def mps_heaviside(interval: QuantizedInterval | IntervalTuple, x0: float = 0.0) -> MPS:
    r"""
    MPS quantization of the Heaviside function :math:`\Theta(x)`.

    Creates an MPS representation of a function

    .. math::
        \Theta(x) = \left\{\begin{array}{ll}
            0, & x < x_0\\
            1, & \mbox{else.}
        \end{array}\right.

    Parameters
    ----------
    interval: QuantizedInterval | IntervalTuple
        The interval over which the function is defined.
    x0 : float
        Position of the step (defaults to 0.0)

    Returns
    -------
    MPS
        The MPS encoding of the function.
    """
    if isinstance(interval, tuple):
        interval = QuantizedInterval(*interval)
    start, step, qubits = interval[0], interval.step, interval.qubits
    output = []
    if x0 < start:
        ndx = 0
    elif x0 > interval[-1]:
        ndx = interval.size
    else:
        ndx = int(math.ceil(x0 - start) / step)
    shift = qubits
    for _ in range(qubits):
        shift -= 1
        bit = (ndx >> shift) & 1
        A = np.zeros((2, 2, 2))
        A[1, :, 1] = 1.0
        if bit == 0:
            A[0, 1, 1] = 1.0
            A[0, 0, 0] = 1.0
        else:
            A[0, 1, 0] = 1.0
        output.append(A)
    output[0] = output[0][[0], :, :]
    output[1] = np.sum(output[1], -1).reshape(2, 2, 1)
    return MPS(output)


def mps_sum_of_exponentials(
    interval: QuantizedInterval | IntervalTuple,
    k: Sequence[float | complex],
    weights: Sequence[float | complex] | complex | float | int = 1.0,
) -> MPS:
    r"""
    Create an MPS representing a sum of exponentials evaluated over a half-open interval.

    Creates an MPS representation of a function

    .. math::
        f(x) = \sum_n w_n exp(-i k_n x)

    Parameters
    ----------
    interval: QuantizedInterval | IntervalTuple
        The interval over which the function is defined.
    k : Sequence[Weight]
        Sequence of exponents in the function.
    weights : Sequence[Weight] | Weight
        Sequence of weights, or a uniform weight for all exponentials (default is 1.0).

    Returns
    -------
    MPS
        The MPS encoding of the function.
    """
    if isinstance(interval, tuple):
        interval = QuantizedInterval(*interval)
    start, step, sites = interval[0], interval.step, interval.qubits
    p = np.asarray(k)
    n = len(p)
    factor = np.exp(p * (start / sites))
    w = np.ones(n) * weights
    if False:
        ndx = list(range(n))
        output = []
        for i in range(sites):
            A = np.zeros((n, 2, n), dtype=np.dtype(type(k[0] * w[0])))
            A[ndx, 0, ndx] = 1.0 * factor
            A[ndx, 1, ndx] = np.exp(p * step * (2 ** (sites - i - 1))) * factor
            output.append(A)
    else:
        phase = (
            (2 ** np.arange(sites - 1, -1, -1).reshape(-1, 1, 1, 1))
            * np.array([0, step]).reshape(1, 2, 1)
            * p
        )
        A = np.exp(phase) * np.eye(n, n).reshape(n, 1, n) * factor
        output = [Ai for Ai in np.ascontiguousarray(A)]
    output[0] = np.sum(output[0], 0).reshape(1, 2, n)
    output[-1] = np.sum(output[-1] * w, -1).reshape(n, 2, 1)
    return MPS(output)


def mps_exponential(interval: QuantizedInterval | IntervalTuple, k: complex = 1) -> MPS:
    """
    Returns an MPS encoding of :math:`exp(k x)`.

    See :func:`mps_sum_of_exponentials`.

    Parameters
    ----------
    interval: QuantizedInterval | IntervalTuple
        The interval over which the function is defined.
    c : complex
        The exponent coefficent (default is 1.0)

    Returns
    -------
    MPS
        An MPS representing the discretized exponential function over the interval.
    """
    return mps_sum_of_exponentials(interval, [k])


def mps_sin(interval: QuantizedInterval | IntervalTuple, k: complex = 1.0) -> MPS:
    """
    Returns an MPS encoding of :math:`sin(k x)`.

    See :func:`mps_sum_of_exponentials`.

    Parameters
    ----------
    interval: QuantizedInterval | IntervalTuple
        The interval over which the function is defined.
    k : complex
        The quasimomentum :math:`k` (default is 1.0)

    Returns
    -------
    MPS
        An MPS representing the discretized sine function over the interval.
    """
    return mps_sum_of_exponentials(interval, [1j * k, -1j * k], [-0.5j, 0.5j])


def mps_cos(interval: QuantizedInterval | IntervalTuple, k: complex = 1.0) -> MPS:
    """
    Returns an MPS encoding of :math:`cos(k x)`.

    See :func:`mps_sum_of_exponentials`.

    Parameters
    ----------
    interval: QuantizedInterval | IntervalTuple
        The interval over which the function is defined.
    c : complex
        The quasimomentum :math:`k` (default is 1.0)

    Returns
    -------
    MPS
        An MPS representing the discretized cosine function over the interval.
    """
    return mps_sum_of_exponentials(interval, [1j * k, -1j * k], [0.5, 0.5])


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
        return _mps_equispaced((start_reg, stop_reg, sites))
    elif isinstance(interval, ChebyshevInterval):
        if interval.endpoints is True:  # Extrema
            start_cheb = 0
            stop_cheb = np.pi + np.pi / (2**sites - 1)
        else:  # Zeros
            start_cheb = np.pi / (2 ** (sites + 1))
            stop_cheb = np.pi + start_cheb
        return mps_affine(
            mps_cos((start_cheb, stop_cheb, sites)),
            (1, -1),  # Reverse order
            (start, stop),
        )
    else:
        raise ValueError(f"Unsupported interval type {type(interval)}")


__all__ = [
    "mps_affine",
    "mps_cos",
    "mps_exponential",
    "mps_heaviside",
    "mps_interval",
    "mps_sin",
    "mps_sum_of_exponentials",
]
