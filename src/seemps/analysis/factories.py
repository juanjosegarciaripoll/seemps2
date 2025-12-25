from __future__ import annotations
import numpy as np
from ..typing import Tensor3, MPSOrder
from ..state import Strategy, MPS, simplify
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


def _map_mps_locations(
    mps_list: list[MPS], mps_order: MPSOrder
) -> list[tuple[int, Tensor3]]:
    """Create a vector that lists which MPS and which tensor is
    associated to which position in the joint Hilbert space.
    """
    tensors = [(0, mps_list[0][0])] * sum(len(mps) for mps in mps_list)
    if mps_order == "A":
        k = 0
        for mps_id, mps in enumerate(mps_list):
            for i, Ai in enumerate(mps):
                tensors[k] = (mps_id, Ai)  # type: ignore
                k += 1
    elif mps_order == "B":
        k = 0
        i = 0
        while k < len(tensors):
            for mps_id, mps in enumerate(mps_list):
                if i < mps.size:
                    tensors[k] = (mps_id, mps[i])  # type: ignore
                    k += 1
            i += 1
    else:
        raise ValueError(f"Invalid mps order {mps_order}")
    return tensors  # type: ignore


def _mps_tensor_terms(mps_list: list[MPS], mps_order: MPSOrder) -> list[MPS]:
    """
    Extends each MPS of a given input list by appending identity tensors to it according
    to the specified MPS order ('A' or 'B'). The resulting list of MPS can be given as terms
    to a tensorized operation between MPS, such as a tensor product or tensor sum.

    Parameters
    ----------
    mps_list : list[MPS]
        The MPS input list.
    mps_order : MPSOrder
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

    mps_map = _map_mps_locations(mps_list, mps_order)
    return [extend_mps(mps_id, mps_map) for mps_id, _ in enumerate(mps_list)]


def mps_tensor_product(
    mps_list: list[MPS],
    mps_order: MPSOrder = "A",
    strategy: Strategy | None = None,
    simplify_steps: bool = False,
) -> MPS:
    """
    Returns the tensor product of a list of MPS, with the sites arranged
    according to the specified MPS order.

    Parameters
    ----------
    mps_list : list[MPS]
        The list of MPS objects to multiply.
    mps_order : MPSOrder, default="A"
        The order in which to arrange the resulting MPS ('A' or 'B').
    strategy : Strategy, optional
        The strategy to use when multiplying the MPS. If None, the tensor product is not simplified.
    simplify_steps : bool, default=False
        Whether to simplify the intermediate steps with `strategy` (if provided)
        or simplify at the end.

    Returns
    -------
    result : MPS | CanonicalMPS
        The resulting MPS from the tensor product of the input list.
    """
    if mps_order == "A":
        nested_sites = [mps._data for mps in mps_list]
        flattened_sites = [site for sites in nested_sites for site in sites]
        result = MPS(flattened_sites)
    elif mps_order == "B":
        terms = _mps_tensor_terms(mps_list, mps_order)
        result = terms[0]
        for _, mps in enumerate(terms[1:]):
            result = (
                simplify(result * mps, strategy=strategy)
                if (strategy and simplify_steps)
                else result * mps
            )
    else:
        raise ValueError(f"Invalid mps order {mps_order}")
    if strategy and not simplify_steps:
        result = simplify(result, strategy=strategy)
    return result


def mps_tensor_sum(
    mps_list: list[MPS],
    mps_order: MPSOrder = "A",
    strategy: Strategy | None = None,
    simplify_steps: bool = False,
) -> MPS:
    """
    Returns the tensor sum of a list of MPS, with the sites arranged
    according to the specified MPS order.

    Parameters
    ----------
    mps_list : list[MPS]
        The list of MPS objects to sum.
    mps_order : MPSOrder, default='A'
        The order in which to arrange the resulting MPS ('A' or 'B').
    strategy : Strategy, optional
        The strategy to use when summing the MPS. If None, the tensor sum is not simplified.
    simplify_steps : bool, default=False
        Whether to simplify the intermediate steps with `strategy` (if provided)
        or simplify at the end.

    Returns
    -------
    result : MPS | CanonicalMPS
        The resulting MPS from the tensor sum of the input list.
    """
    if mps_order == "A":
        result = _mps_tensor_sum_serial_order(mps_list)
    elif mps_order == "B":
        terms = _mps_tensor_terms(mps_list, mps_order)
        result = terms[0]
        for _, mps in enumerate(terms[1:]):
            result = (
                simplify(result + mps, strategy=strategy)
                if (strategy and simplify_steps)
                else (result + mps).join()
            )
    else:
        raise ValueError(f"Invalid mps order {mps_order}")
    if strategy and not simplify_steps:
        result = simplify(result, strategy=strategy)
    return result


def _mps_tensor_sum_serial_order(mps_list: list[MPS]) -> MPS:
    """
    Computes the MPS tensor sum in serial order in an optimized manner.
    """

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
        for _, A in enumerate(mps_list)
        for i, Ai in enumerate(A)
    ]
    output[0] = output[0][[0], :, :]
    output[-1] = output[-1][:, :, [1]]
    return MPS(output)
