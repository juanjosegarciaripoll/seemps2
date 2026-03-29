from __future__ import annotations
import numpy as np
from ..typing import Vector, Operator
from ..state import Strategy, DEFAULT_STRATEGY
from ..operators import MPOList, MPO
from ..operators.projectors import identity_mpo


def qubo_mpo(
    J: Operator | None = None,
    h: Vector | None = None,
    strategy: Strategy = DEFAULT_STRATEGY,
) -> MPO:
    """Return the MPO associated to a QUBO operator.

    The operator is defined according to the mathematical notation
    :math:`\\sum_i J_{ij} s_i s_j + \\sum_i h_i s_i`,
    with the matrix of interactions 'J' and the vector of local
    fields 'h'. At least one of these must be provided.

    Parameters
    ----------
    J : Operator | None
        Matrix of Ising coupling between qubits (Default value = None)
    h : Vector | None :
        Vector of local magnetic fields (Default value = None)
    strategy : Strategy, default = DEFAULT_STRATEGY
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
            raise Exception("Must provide either J or h")
        #
        data = []
        id2 = np.eye(2)
        for i, hi in enumerate(h):
            A = np.zeros((2, 2, 2, 2), dtype=hi.dtype)
            A[0, 1, 1, 1] = hi
            A[1, :, :, 1] = id2
            A[0, :, :, 0] = id2
            data.append(A)
        data[-1] = data[-1][:, :, :, [1]]
        data[0] = data[0][[0], :, :, :]
    else:
        Jmatrix = np.asarray(J)
        if h is not None:
            Jmatrix += np.diag(h)
        L = len(Jmatrix)
        id2 = np.eye(2)
        data = []
        for i in range(L):
            A = np.zeros((i + 2, 2, 2, i + 3))
            A[0, 1, 1, 1] = Jmatrix[i, i]
            A[1, :, :, 1] = np.eye(2)
            A[0, :, :, 0] = np.eye(2)
            A[0, 1, 1, i + 2] = 1.0
            for j in range(i):
                A[j + 2, 1, 1, 1] = Jmatrix[i, j] + Jmatrix[j, i]
                A[j + 2, :, :, j + 2] = np.eye(2)
            data.append(A)
        data[-1] = data[-1][:, :, :, [1]]
        data[0] = data[0][[0], :, :, :]
    return MPO(data, strategy)


def qubo_exponential_mpo(
    J: Operator | None = None,
    h: Vector | None = None,
    beta: float = -1.0,
    strategy: Strategy = DEFAULT_STRATEGY,
) -> MPO | MPOList:
    """Return the MPO associated to the exponential $\\exp(\\beta H)$ of
    a QUBO operator.

    The QUBO operator is defined as a sum of
    longitudinal couplings and magnetic fields
    :math:`H = \\sum_i J_{ij} s_i s_j + \\sum_i h_i s_i`
    described by the interaction matrix `J` and the vector field `h`.
    At least one of these must be provided.

    Parameters
    ----------
    J : Operator | None
        Matrix of Ising coupling between qubits (Default value = None)
    h : Vector | None :
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
            raise Exception("Must provide either J or h")
        #
        dtype = np.result_type(h, beta)
        data = []
        for i, hi in enumerate(h):
            A = np.zeros((1, 2, 2, 1), dtype=dtype)
            A[0, 1, 1, 0] = np.exp(beta * hi)
            A[0, 0, 0, 0] = 1.0
            data.append(A)
        return MPO(data, strategy)
    else:
        Jmatrix = np.asarray(J)
        L = len(Jmatrix)
        dtype = np.result_type(Jmatrix, beta)
        if h is not None:
            dtype = np.result_type(dtype, h)
        id2 = np.eye(2, dtype=dtype)
        noop = id2.reshape(1, 2, 2, 1)
        out = []
        local_terms = np.diag(Jmatrix).copy()
        if h is not None:
            local_terms = local_terms + np.asarray(h)

        for i, Ji in enumerate(local_terms):
            couplings = [Jmatrix[i, j] + Jmatrix[j, i] for j in range(i + 1, L)]
            if Ji == 0 and all(Jij == 0 for Jij in couplings):
                continue
            data = [noop] * i
            if i == L - 1:
                A = np.zeros((1, 2, 2, 1), dtype=dtype)
                A[0, 0, 0, 0] = 1.0
                A[0, 1, 1, 0] = np.exp(beta * Ji)
                data.append(A)
            else:
                A = np.zeros((1, 2, 2, 2), dtype=dtype)
                A[0, 0, 0, 0] = 1.0
                A[0, 1, 1, 1] = np.exp(beta * Ji)
                data.append(A)
                for Jij in couplings[:-1]:
                    B = np.zeros((2, 2, 2, 2), dtype=dtype)
                    B[0, :, :, 0] = id2
                    B[1, 0, 0, 1] = 1.0
                    B[1, 1, 1, 1] = np.exp(beta * Jij)
                    data.append(B)
                C = np.zeros((2, 2, 2, 1), dtype=dtype)
                C[0, :, :, 0] = id2
                C[1, 0, 0, 0] = 1.0
                C[1, 1, 1, 0] = np.exp(beta * couplings[-1])
                data.append(C)
            out.append(MPO(data, strategy))

        if not out:
            out = [identity_mpo([2] * L, strategy)]
        return MPOList(out)
