from __future__ import annotations
import numpy as np
from typing import TypeVar, Union
from ..typing import Tensor3
from ..state import (
    MPS,
    MPSSum,
    Strategy,
    DEFAULT_STRATEGY,
    Truncation,
    DEFAULT_TOLERANCE,
    Simplification,
)
from ..truncate import simplify
from .mesh import (
    Interval,
    RegularClosedInterval,
    RegularHalfOpenInterval,
    ChebyshevZerosInterval,
)

DEFAULT_FACTORY_STRATEGY = Strategy(
    method=Truncation.RELATIVE_SINGULAR_VALUE,
    tolerance=DEFAULT_TOLERANCE,
    simplify=Simplification.VARIATIONAL,
    simplification_tolerance=DEFAULT_TOLERANCE,
    normalize=False,
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
    for idx, tensor in enumerate(tensors_bulk):
        tensor[0, 0, 0] = 1
        tensor[0, 1, 0] = np.exp(c * step * 2 ** (sites - (idx + 2)))
    tensors = [tensor_1] + tensors_bulk + [tensor_2]
    return MPS(tensors)


def mps_sin(
    start: float, stop: float, sites: int, strategy: Strategy = DEFAULT_FACTORY_STRATEGY
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
    strategy : Strategy, default = DEFAULT_STRATEGY
        The MPS simplification strategy to apply.

    Returns
    -------
    MPS
        An MPS representing the discretized sine function over the interval.
    """
    mps_1 = mps_exponential(start, stop, sites, c=1j)
    mps_2 = mps_exponential(start, stop, sites, c=-1j)

    return simplify(-0.5j * (mps_1 - mps_2), strategy=strategy)


def mps_cos(
    start: float, stop: float, sites: int, strategy: Strategy = DEFAULT_FACTORY_STRATEGY
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
    strategy : Strategy, default = DEFAULT_STRATEGY
        The MPS simplification strategy to apply.

    Returns
    -------
    MPS
        An MPS representing the discretized cosine function over the interval.
    """
    mps_1 = mps_exponential(start, stop, sites, c=1j)
    mps_2 = mps_exponential(start, stop, sites, c=-1j)

    return simplify(0.5 * (mps_1 + mps_2), strategy=strategy)


_State = TypeVar("_State", bound=Union[MPS, MPSSum])


def mps_affine_transformation(mps: _State, orig: tuple, dest: tuple) -> _State:
    """
    Applies an affine transformation to an MPS, mapping it from one interval [x0, x1] to another [u0, u1].
    This is a transformation u = a * x + b, with u0 = a * x0 + b and and  u1 = a * x1 + b.
    Hence, a = (u1 - u0) / (x1 - x0) and b = ((u1 + u0) - a * (x0 + x1)) / 2.

    Parameters
    ----------
    mps : MPS
        The MPS to be transformed.
    orig : tuple
        A tuple (x0, x1) representing the original interval.
    dest : tuple
        A tuple (u0, u1) representing the destination interval.

    Returns
    -------
    MPS
        The transformed MPS.
    """
    x0, x1 = orig
    u0, u1 = dest
    a = (u1 - u0) / (x1 - x0)
    b = 0.5 * ((u1 + u0) - a * (x0 + x1))
    mps_affine = a * mps
    if abs(b) > np.finfo(np.float64).eps:
        I = MPS([np.ones((1, 2, 1))] * len(mps_affine))
        mps_affine = mps_affine + b * I
        # Preserve the input type
        if isinstance(mps, MPS):
            return mps_affine.join()
    return mps_affine


def mps_interval(interval: Interval, strategy: Strategy = DEFAULT_FACTORY_STRATEGY):
    """
    Returns an MPS corresponding to a specific type of interval (open, closed, or Chebyshev zeros).

    Parameters
    ----------
    interval : Interval
        The interval object containing start and stop points and the interval type.
    strategy : Strategy, default = DEFAULT_STRATEGY
        The MPS simplification strategy to apply.

    Returns
    -------
    mps : MPS
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
        start_zeros = np.pi / (2 ** (sites + 1))
        stop_zeros = np.pi + start_zeros
        mps_zeros = -1.0 * mps_cos(start_zeros, stop_zeros, sites, strategy=strategy)
        return mps_affine_transformation(mps_zeros, (-1, 1), (start, stop))
    else:
        raise ValueError(f"Unsupported interval type {type(interval)}")


def map_mps_locations(mps_list: list[MPS], mps_order: str) -> list[tuple[int, Tensor3]]:
    """Create a vector that lists which MPS and which tensor is
    associated to which position in the joint Hilbert space.
    """
    tensors = [(0, mps_list[0][0])] * sum(len(mps) for mps in mps_list)
    if mps_order == "A":
        k = 0
        for mps_id, mps in enumerate(mps_list):
            for i, Ai in enumerate(mps):
                tensors[k] = (mps_id, Ai)
                k += 1
    elif mps_order == "B":
        k = 0
        i = 0
        while k < len(tensors):
            for mps_id, mps in enumerate(mps_list):
                if i < mps.size:
                    tensors[k] = (mps_id, mps[i])
                    k += 1
            i += 1
    else:
        raise ValueError(f"Invalid mps order {mps_order}")
    return tensors


def mps_tensor_terms(mps_list: list[MPS], mps_order: str) -> list[MPS]:
    """
    Extends each MPS of a given input list by appending identity tensors to it
    according to the specified MPS order ('A' or 'B').
    The resulting list of MPS can be given as terms to a tensorized operation between MPS,
    such as a tensor product or tensor sum.

    Parameters
    ----------
    mps_list : list[MPS]
        The MPS input list.
    mps_order : str
        The order in which to arrange the qubits for each resulting MPS term ('A' or 'B').

    Returns
    -------
    list[MPS]
        The resulting list of MPS terms.
    """

    def extend_mps(mps_id: int, mps_map: list[tuple[int, Tensor3]]) -> MPS:
        D = 1
        output = [mps_map[0][1]] * len(mps_map)
        for k, (site_mps, site_tensor) in enumerate(mps_map):
            if mps_id == site_mps:
                output[k] = site_tensor
                D = site_tensor.shape[-1]
            else:
                site_dimension = site_tensor.shape[1]
                output[k] = np.eye(D).reshape(D, 1, D) * np.ones((1, site_dimension, 1))
        return MPS(output)

    mps_map = map_mps_locations(mps_list, mps_order)
    return [extend_mps(mps_id, mps_map) for mps_id, _ in enumerate(mps_list)]


def mps_tensor_product(
    mps_list: list[MPS],
    mps_order: str = "A",
    strategy: Strategy = DEFAULT_FACTORY_STRATEGY,
) -> MPS:
    """
    Returns the tensor product of a list of MPS, with the sites arranged
    according to the specified MPS order.

    Parameters
    ----------
    mps_list : list[MPS]
        The list of MPS objects to multiply.
    mps_order : str
        The order in which to arrange the resulting MPS ('A' or 'B').
    strategy : optional
        The strategy to use when multiplying the MPS.

    Returns
    -------
    MPS
        The resulting MPS from the tensor product of the input list.
    """
    if mps_order == "A":
        result = MPS([site for mps in mps_list for site in mps])
    else:
        terms = mps_tensor_terms(mps_list, mps_order)
        result = terms[0]
        for idx, mps in enumerate(terms[1:]):
            result = result * mps
    return simplify(result, strategy=strategy)


def mps_tensor_sum(
    mps_list: list[MPS], mps_order: str = "A", strategy: Strategy = DEFAULT_STRATEGY
) -> MPS:
    """
    Returns the tensor sum of a list of MPS, with the sites arranged
    according to the specified MPS order.

    Parameters
    ----------
    mps_list : list[MPS]
        The list of MPS objects to sum.
    mps_order : str
        The order in which to arrange the resulting MPS ('A' or 'B').
    strategy : optional
        The strategy to use when summing the MPS.

    Returns
    -------
    MPS
        The resulting MPS from the tensor sum of the input list.
    """
    if mps_order == "A":
        result = _mps_tensor_sum_serial_order(mps_list)
    else:
        result = MPSSum(
            [1.0] * len(mps_list), mps_tensor_terms(mps_list, mps_order)
        ).join()
    if strategy.get_simplify_flag():
        return simplify(result, strategy=strategy)
    return result


def _mps_tensor_sum_serial_order(mps_list: list[MPS]) -> MPS:
    def extend_tensor(A: Tensor3, first: bool, last: bool) -> Tensor3:
        a, d, b = A.shape
        output = np.zeros((a + 2, d, b + 2), dtype=A.dtype)
        output[0, :, 0] = np.ones(d)  # No MPS applied
        output[1, :, 1] = np.ones(d)  # One MPS applied
        if first:
            if last:
                output[[0], :, [1]] = A
            else:
                output[[0], :, 2:] = A
        elif last:
            output[2:, :, [1]] = A
        else:
            output[2:, :, 2:] = A
        return output

    output = [
        extend_tensor(Ai, i == 0, i == len(A) - 1)
        for n, A in enumerate(mps_list)
        for i, Ai in enumerate(A)
    ]
    output[0] = output[0][[0], :, :]
    output[-1] = output[-1][:, :, [1]]
    return MPS(output)
