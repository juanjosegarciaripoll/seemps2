import numpy as np
from ..typing import *
from ..operators import MPOList, MPO


def qubo_mpo(
    J: Optional[Operator] = None, h: Optional[Vector] = None, **kwdargs
) -> MPO:
    """Return the MPO associated to a QUBO operator.

    The operator is defined according to the mathematical notation
    :math:`\\sum_i J_{ij} s_i s_j + \\sum_i h_i s_i`,
    with the matrix of interactions 'J' and the vector of local
    fields 'h'. At least one of these must be provided.

    Parameters
    ----------
    J : Optional[Operator]
        Matrix of Ising coupling between qubits (Default value = None)
    h : Optional[Vector] :
        Vector of local magnetic fields (Default value = None)
    **kwdargs :
        Other arguments accepted by :class:`MPO`

    Returns
    -------
    MPO
        Matrix-product operator implementing this Hamiltonian
    """
    if J is None:
        #
        # Just magnetic field. A much simpler operator
        if h is None:
            raise Exception("In QUBO_MPO, must provide either J or h")
        #
        data = []
        id2 = np.eye(2)
        for i, hi in enumerate(h):
            A = np.zeros((2, 2, 2, 2), dtype=hi.dtype)
            A[0, 1, 1, 1] = hi
            A[1, :, :, 1] = id2
            A[0, :, :, 0] = id2
            data.append(A)
        A = A[:, :, :, [1]]
        data[-1] = A
        data[0] = data[0][[0], :, :, :]
    else:
        if h is not None:
            J = J + np.diag(h)
        L = len(J)
        id2 = np.eye(2)
        data = []
        for i in range(L):
            A = np.zeros((i + 2, 2, 2, i + 3))
            A[0, 1, 1, 1] = J[i, i]
            A[1, :, :, 1] = np.eye(2)
            A[0, :, :, 0] = np.eye(2)
            A[0, 1, 1, i + 2] = 1.0
            for j in range(i):
                A[j + 2, 1, 1, 1] = J[i, j] + J[j, i]
                A[j + 2, :, :, j + 2] = np.eye(2)
            data.append(A)
        data[-1] = data[-1][:, :, :, [1]]
        data[0] = data[0][[0], :, :, :]
    return MPO(data, **kwdargs)


def qubo_exponential_mpo(
    J: Optional[Operator] = None,
    h: Optional[Vector] = None,
    beta: float = -1.0,
    **kwdargs
) -> Union[MPO, MPOList]:
    """Return the MPO associated to the exponential $\\exp(\\beta H)$ of
    a QUBO operator.

    The QUBO operator is defined as a sum of
    longitudinal couplings and magnetic fields
    :math:`H = \\sum_i J_{ij} s_i s_j + \\sum_i h_i s_i`
    described by the interaction matrix `J` and the vector field `h`.
    At least one of these must be provided.

    Parameters
    ----------
    J : Optional[Operator]
        Matrix of Ising coupling between qubits (Default value = None)
    h : Optional[Vector] :
        Vector of local magnetic fields (Default value = None)
    beta : float :
        Exponential prefactor (Default value = -1.0)
    **kwdargs :
        Other arguments accepted by :class:`MPO`

    Returns
    -------
    MPOList | MPO
        MPO or set of them to implement the exponential of an Ising interaction.
    """
    if J is None:
        #
        # Just magnetic field. A much simpler operator
        if h is None:
            raise Exception("In QUBO_MPO, must provide either J or h")
        #
        data = []
        for i, hi in enumerate(h):
            A = np.zeros((1, 2, 2, 1))
            A[0, 1, 1, 1] = np.exp(beta * hi)
            A[0, 0, 0, 0] = 1.0
            data.append(A)
        return MPO(data, **kwdargs)
    else:
        if h is not None:
            J = J + np.diag(h)
        J = (J + J.T) / 2
        L = len(J)
        noop = np.eye(2).reshape(1, 2, 2, 1)
        out = []
        for i in range(L):
            data = [noop] * i
            A = np.zeros((1, 2, 2, 2))
            A[0, 1, 1, 1] = np.exp(beta * J[i, i])
            A[0, 0, 0, 0] = 1.0
            for j in range(i + 1, L):
                A = np.zeros((2, 2, 2, 2))
                A[1, 1, 1, 1] = np.exp(beta * J[i, j])
                A[1, 0, 0, 1] = 1.0
                A[0, 0, 0, 0] = 1.0
                A[0, 1, 1, 0] = 1.0
                data.append(A)
            data[-1] = A[:, :, :, [0]] + A[:, :, :, [1]]
            out.append(MPO(data, **kwdargs))
        return MPOList(out)
