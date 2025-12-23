from __future__ import annotations
import numpy as np
from collections.abc import Iterable
from ..typing import VectorLike, Tensor4
from ..state import Strategy, DEFAULT_STRATEGY
from ..operators import MPO


def mpo_weighted_shifts(
    L: int,
    weights: VectorLike,
    shifts: tuple[int, int] | list[int],
    periodic: bool = False,
    base: int = 2,
) -> MPO:
    r"""Return an MPO that implements a linear combination of arithmetic displacements.

    This function creates a matrix product operator that implements

    .. math::
       O |s\rangle \to \sum_i w_i |s + i\rangle

    The operator is very useful to implement finite difference approximations.

    Parameters
    ----------
    L : int
        Number of qubits in the quantum register
    weights : VectorLike
        List or array of weights to apply to each displacement
    shifts : list[int] | tuple[int,int]
        Range of displacements to be applied
    periodic : bool, default = False
        Whether to wrap around integers when shifting.
    base : int, default = 2
        Encoding of the quantum register elements. By default `base=2`
        meaning we work with qubits.

    Returns
    -------
    MPO
        Matrix product operator for `O` above.
    """
    O = mpo_shifts(L, shifts, periodic, base)
    O[L - 1] = np.einsum("aijb,bc->aijc", O[-1], np.reshape(weights, (-1, 1)))
    return O


def mpo_shifts(
    L: int, shifts: tuple[int, int] | list[int], periodic: bool = False, base: int = 2
) -> MPO:
    r"""Return an MPO with a free index, capable of displacing a quantum
    register by a range of integers.

    This function creates matrix product operators that add a specific integer
    to a quantum register encoded in an MPS. The `shifts` can be a list of those
    integers, or a tuple denoting a range, converted as
    `shifts=list(range(*shifts))`.

    The MPO is a bit special, in that the last tensor of the operator, say
    `A[L]` will have a final index with size `M` equal to the number of shifts.

    Parameters
    ----------
    L : int
        Number of qubits in the quantum register
    shifts : list[int] | tuple[int,int]
        Range of displacements to be applied
    periodic : bool, default = False
        Whether to wrap around integers when shifting.
    base : int, default = 2
        Encoding of the quantum register elements. By default `base=2`
        meaning we work with qubits.

    Returns
    -------
    MPO
        Parameterized MPO with an extra bond at the end
    """
    if isinstance(shifts, tuple):
        r = np.arange(shifts[0], shifts[1], dtype=int)
    else:
        r = np.asarray(shifts, dtype=int)
    tensors: list[Tensor4] = []
    bits = np.arange(base).reshape(base, 1)
    for _ in reversed(range(L)):
        #
        # The shift r[j] adds to the current bit s[i], producing
        # an integer r[j] + s[i] = 2 * r' + s[i]', with a new
        # value of the bit s[i]' and a carry displacement r' for
        # the new site
        #
        # These are the carry-on values
        newr_matrix = (bits + r) // base
        # These are the indices that the qudit states are mapped to
        news_matrix = (bits + r) % base
        # A vector with all carry-on values
        newr = np.sort(np.unique(newr_matrix.reshape(-1)))
        # and a matrix with the positions of the carry-on values into
        # this vector. This transforms carry-on-values to bond dimension
        # indices
        newr_matrix = np.searchsorted(newr, newr_matrix)
        A = np.zeros((newr.size, base, base, r.size))
        A[newr_matrix, news_matrix, bits, np.arange(r.size)] = 1.0
        tensors.append(A)
        r = newr
    A = tensors[-1]
    if periodic:
        tensors[-1] = np.sum(A, 0).reshape((1,) + A.shape[1:])
    else:
        ndx = np.nonzero(r == 0)[0]
        if ndx.size:
            tensors[-1] = A[ndx, :, :, :]
        else:
            tensors[-1] = A[[0], :, :, :] * 0.0
    return MPO(list(reversed(tensors)))


def twoscomplement(
    L: int,
    control: int = 0,
    sites: Iterable[int] | None = None,
    strategy: Strategy = DEFAULT_STRATEGY,
) -> MPO:
    """Return an MPO that performs a two's complement of the selected qubits
    depending on a 'control' qubit in a register with L qubits.

    Parameters
    ----------
    L : int :
        Real size of register
    control : int :
        Which qubit (relative to sites) controls the sign. (Default value = 0)
    sites : Iterable[int] | None :
        The qubits involved in the MPO. (Default value = L)
    **kwdargs :
        Arguments for :meth:`MPO.__init__`
    """

    if sites is not None:
        sites = sorted(sites)
        out = twoscomplement(
            len(sites), control=sites.index(control), sites=None, strategy=strategy
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
        return MPO(data, strategy)
