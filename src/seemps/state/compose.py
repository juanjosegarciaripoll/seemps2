from __future__ import annotations

import numpy as np
from .mps import MPS
from ..typing import Tensor3, MPSOrder
from .simplification import simplify_mps, Strategy

# TODO: All this logic *must* be simplified


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
        result = MPS(sum((mps._data for mps in mps_list), []))
    elif mps_order == "B":
        terms = _mps_tensor_terms(mps_list, mps_order)
        result = terms[0]
        for _, mps in enumerate(terms[1:]):
            result = (
                simplify_mps(result * mps, strategy=strategy)
                if (strategy and simplify_steps)
                else result * mps
            )
    else:
        raise ValueError(f"Invalid mps order {mps_order}")
    if strategy and not simplify_steps:
        result = simplify_mps(result, strategy=strategy)
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
    if len(mps_list) == 1:
        return mps_list[0]
    if mps_order == "A":
        result = _mps_tensor_sum_serial_order(mps_list)
    elif mps_order == "B":
        terms = _mps_tensor_terms(mps_list, mps_order)
        result = terms[0]
        for _, mps in enumerate(terms[1:]):
            result = (
                simplify_mps(result + mps, strategy=strategy)
                if (strategy and simplify_steps)
                else (result + mps).join()
            )
    else:
        raise ValueError(f"Invalid mps order {mps_order}")
    if strategy and not simplify_steps:
        result = simplify_mps(result, strategy=strategy)
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
