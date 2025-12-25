from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from ..typing import Unitary, Tensor3, Tensor4

# Python versions of some contractions that are implemented
# in Cython in contractions.pxi Only here for backup.


def _contract_nrjl_ijk_klm(U: Unitary, A: Tensor3, B: Tensor3) -> Tensor4:
    #
    # Assuming U[n*r,j*l], A[i,j,k] and B[k,l,m]
    # Implements np.einsum('ijk,klm,nrjl -> inrm', A, B, U)
    # See tests.test_contractions for other implementations and timing
    #
    a, d, b = A.shape
    b, e, c = B.shape
    return np.matmul(
        U, np.matmul(A.reshape(-1, b), B.reshape(b, -1)).reshape(a, -1, c)
    ).reshape(a, d, e, c)


def _contract_last_and_first(A: NDArray, B: NDArray) -> NDArray:
    """Contract last index of `A` and first from `B`"""
    sA = A.shape
    sB = B.shape
    return np.matmul(A, B.reshape(sB[0], -1)).reshape(sA[:-1] + sB[1:])


__all__ = ["_contract_nrjl_ijk_klm", "_contract_last_and_first"]
