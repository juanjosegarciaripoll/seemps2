import numpy as np
from ..typing import *
from ..operators import MPO


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
