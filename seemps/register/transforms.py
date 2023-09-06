import numpy as np
from ..typing import *
from ..state import MPS
from ..mpo import MPO
from .. import truncate


def wavefunction_product(
    ψ: MPS, ξ: MPS, conjugate: bool = False, simplify: bool = True, **kwdargs
) -> MPS:
    """Implement a nonlinear transformation that multiplies two MPS, to
    create a new MPS with combined bond dimensions. In other words, act
    with the nonlinear transformation <s|ψξ> = ψ(s)ξ(s)|s> or
    <s|ψ*ξ> = ψ*(s)ξ(s)|s>

    Parameters
    ----------
    ψ :
        ξ
    conjugate :
        Conjugate ψ or not
    simplify :
        Simplify the state afterwards or not
    kwdargs :
        Arguments to simplify
    Output :

    mps :
        The MPS product ψξ or ψ
    ψ : MPS :

    ξ : MPS :

    conjugate : bool :
        (Default value = False)
    simplify : bool :
        (Default value = True)
    **kwdargs :

    ψ: MPS :

    ξ: MPS :

    conjugate: bool :
         (Default value = False)
    simplify: bool :
         (Default value = True)

    Returns
    -------


    """

    def combine(A, B):
        """

        Parameters
        ----------
        A :

        B :


        Returns
        -------


        """
        # Combine both tensors
        a, d, b = A.shape
        c, d, e = B.shape
        if conjugate:
            A = A.conj()
        D = np.array(
            [np.outer(A[:, i, :].reshape(-1), B[:, i, :].reshape(-1)) for i in range(d)]
        )
        D = np.einsum("iabce->acibe", np.array(D).reshape(d, a, b, c, e)).reshape(
            a * c, d, b * e
        )
        return D

    out = MPS([combine(A, B) for A, B in zip(ψ, ξ)])
    if simplify:
        out, _, _ = truncate.simplify(out, **kwdargs)
    return out


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
