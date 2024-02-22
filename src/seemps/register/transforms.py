from __future__ import annotations
import numpy as np
from typing import Optional, Iterable
from ..mpo import MPO


def mpo_shifts(
    L: int, shifts: tuple[int, int] | list[int], periodic: bool = False
) -> MPO:
    r"""Return an MPO with a free index, capable of displacing a quantum
    register by a range of integers.

    This function creates matrix product operators that add a specific integer
    to a quantum register encoded in an MPS. The `shifts` can be a list of those
    integers, or a tuple denoting a range, converted as
    `shifts=list(range(*shifts))`.

    The MPO is a bit special, in that the last tensor of the operator, say
    `A[L]` will have a final index with size `M` equal to the number of shifts.

    Arguments
    ---------
    L : int
        Number of qubits in the quantum register
    shifts : list[int] | tuple[int,int]
        Range of displacements to be applied
    periodic : bool, default = False
        Whether to wrap around integers when shifting.

    Returns
    -------
    MPO
        Parameterized MPO with an extra bond at the end
    """
    if isinstance(shifts, tuple):
        r = np.arange(shifts[0], shifts[1], dtype=int)
    else:
        r = np.asarray(shifts, dtype=int)
    tensors = []
    bits = np.reshape([0, 1], (2, 1))
    for i in reversed(range(L)):
        #
        # The shift r[j] adds to the current bit s[i], producing
        # an integer r[j] + s[i] = 2 * r' + s[i]', with a new
        # value of the bit s[i]' and a carry displacement r' for
        # the new site
        #
        newr = (bits + r) >> 1
        news = (bits + r) & 1
        newr_ndx = np.sort(np.unique(newr.reshape(-1)))
        b = r.size
        a = newr_ndx.size
        A = np.zeros((a, 2, 2, b))
        for s in (0, 1):
            for a, ra in enumerate(r):
                A[newr[s, a] - newr_ndx[0], news[s, a], s, a] = 1.0
        tensors.append(A)
        r = newr_ndx
    if periodic:
        tensors[-1] = np.sum(A, 0).reshape((1,) + A.shape[1:])
    else:
        ndx = np.argmin(np.abs(r))
        if r[ndx] == 0:
            tensors[-1] = A[[ndx], :, :, :]
        else:
            tensors[-1] = A[[0], :, :, :] * 0.0
    return MPO(list(reversed(tensors)))


def twoscomplement(
    L: int, control: int = 0, sites: Optional[Iterable[int]] = None, **kwdargs
) -> MPO:
    """Return an MPO that performs a two's complement of the selected qubits
    depending on a 'control' qubit in a register with L qubits.

    Arguments
    ---------
    L       -- Real size of register
    control -- Which qubit (relative to sites) controls the sign.
               Defaults to the first qubit in 'sites'.
    sites   -- The qubits involved in the MPO. Defaults to range(L).
    kwdargs -- Arguments for MPO.

    Parameters
    ----------
    L : int :

    control : int :
        (Default value = 0)
    sites : Optional[Iterable[int]] :
        (Default value = None)
    **kwdargs :

    L: int :

    control: int :
         (Default value = 0)
    sites: Optional[Iterable[int]] :
         (Default value = None)

    Returns
    -------


    """

    if sites is not None:
        sites = sorted(sites)
        out = twoscomplement(
            len(sites), control=sites.index(control), sites=None, **kwdargs
        )
        return out.extend(L, sites=sites)
    else:
        A0 = np.zeros((2, 2, 2, 2))
        A0[0, 0, 0, 0] = 1.0
        A0[1, 1, 1, 1] = 1.0
        A = np.zeros((2, 2, 2, 2))
        A[0, 0, 0, 0] = 1.0
        A[0, 1, 1, 0] = 1.0
        A[1, 1, 0, 1] = 1.0
        A[1, 0, 1, 1] = 1.0
        data = [A0 if i == control else A for i in range(L)]
        A = data[0]
        data[0] = A[[0], :, :, :] + A[[1], :, :, :]
        A = data[-1]
        data[-1] = A[:, :, :, [0]] + A[:, :, :, [1]]
        return MPO(data, **kwdargs)
