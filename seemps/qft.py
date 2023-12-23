import numpy as np
from numpy import pi as π
from .typing import *
from .state import MPS, MPSSum
from .operators import MPO, MPOList


def qft_mpo(N: int, sign: int = -1, **kwargs) -> MPOList:
    """Create an MPOList object representing a Quantum Fourier Transform
    for a quantum register with `N` qubits.

    Parameters
    ----------
    N : int
        Number of qubits in the MPO.
    sign : int, default = -1
        Sign (+1 or -1) in the exponent of the transform. Defaults to
        the sign of the direct quantum Fourier transform.
    **kwargs :
        Other arguments accepted by :class:`MPO`

    Returns
    -------
    MPOList
        A sequence of :class:`MPO` that implements the transform.
    """

    def fix_last(mpo_list):
        A = mpo_list[-1]
        shape = A.shape
        A = np.sum(A, -1).reshape(shape[0], shape[1], shape[2], 1)
        return mpo_list[:-1] + [A]

    # Tensor doing nothing
    noop = np.eye(2).reshape(1, 2, 2, 1)
    #
    # Beginning Hadamard
    H = np.array([[1, 1], [1, -1]]) / np.sqrt(2.0)
    Hop = np.zeros((1, 2, 2, 2))
    Hop[0, 1, :, 1] = H[1, :]
    Hop[0, 0, :, 0] = H[0, :]
    #
    # Conditional rotations
    R0 = np.zeros((2, 2, 2, 2))
    R0[0, 0, 0, 0] = 1.0
    R0[0, 1, 1, 0] = 1.0
    R0[1, 0, 0, 1] = 1.0
    R1 = np.zeros((2, 2, 2, 2))
    R1[1, 1, 1, 1] = 1.0
    jϕ = sign * 1j * π
    rots = [R0 + R1 * np.exp(jϕ / (2**n)) for n in range(1, N)]
    #
    return MPOList(
        [
            MPO(fix_last([noop] * n + [Hop] + rots[: N - n - 1]), **kwargs)
            for n in range(0, N)
        ],
        **kwargs
    )


def iqft_mpo(N: int, **kwargs) -> MPOList:
    """:class:`MPOList` implementing the inverse quantum Fourier transform.

    Parameters
    ----------
    N : int
        Number of qubits in the MPO.
    **kwargs :
        Other arguments accepted by :class:`MPO`


    Returns
    -------
    MPOList
        A sequence of :class:`MPO` that implements the transform.
    """
    return qft_mpo(N, +1, **kwargs)


def qft(state: Union[MPS, MPSSum], **kwargs) -> Union[MPS, MPSSum]:
    """Apply the quantum Fourier transform onto a quantum register
    of qubits encoded in the matrix-product 'state'.

    Parameters
    ----------
    state : MPS
        Quantum register to transform
    **kwargs :
        Arguments accepted by :class:`MPO`

    Returns
    -------
    MPS
        Transformed quantum state after application of operators.
    """
    return qft_mpo(state.size, sign=-1, **kwargs).apply(state)


def iqft(state: Union[MPS, MPSSum], **kwargs) -> Union[MPS, MPSSum]:
    """Apply the inverse quantum Fourier transform onto a quantum register
    of qubits encoded in the matrix-product 'state'.

    Parameters
    ----------
    state : MPS
        Quantum register to transform
    **kwargs :
        Arguments accepted by :class:`MPO`

    Returns
    -------
    MPS
        Transformed quantum state after application of operators.
    """
    return qft_mpo(state.size, sign=+1, **kwargs).apply(state)


def qft_flip(state: Union[MPS, MPSSum]) -> Union[MPS, MPSSum]:
    """Swap the qubits in the quantum register, to fix the reversal
    suffered during the quantum Fourier transform.

    Parameters
    ----------
    state : MPS
        Transformed state

    Returns
    -------
    MPS
        State with qubits reversed.
    """
    if isinstance(state, MPSSum):
        return MPSSum(state.weights, [qft_flip(s) for s in state.states])  # type: ignore
    return MPS(
        [np.moveaxis(A, [0, 1, 2], [2, 1, 0]) for A in reversed(state)],
        error=state.error(),
    )


def qft_wavefunction(Ψ: Vector) -> Vector:
    """Implement the QFT on a state vector.

    This routine uses :func:`numpy.fft.fft` and is provided here as a
    convenience, so that the user can compare the action of the QFT on an
    MPS and on a vector.

    Parameters
    ----------
    Ψ : Vector
        A wavefunction in :class:`numpy.ndarray` format.

    Returns
    -------
    Vector
        Transformed state.
    """
    return np.fft.fft(Ψ) / np.sqrt(Ψ.size)


def qft_nd_mpo(
    sites: list[int], N: Optional[int] = None, sign: int = -1, **kwargs
) -> MPOList:
    """Create an MPOList object representing a Quantum Fourier Transform
    for subset of qubits in a quantum register with `N` qubits.

    Parameters
    ----------
    sites : list[int]
        List of qubits on which the transform acts
    N : int, default = len(sites)
        Number of qubits in the register.
    sign : int, default = -1
        Sign (+1 or -1) in the exponent of the transform. Defaults to
        the sign of the direct quantum Fourier transform.
    **kwargs :
        Other arguments accepted by :class:`MPO`

    Returns
    -------
    MPOList
        A sequence of :class:`MPO` that implements the transform.
    """
    if N is None:
        N = max(sites) + 1
    #
    # Construct a bare transformation that does nothing
    small_noop = np.eye(2).reshape(1, 2, 2, 1)
    noop = np.eye(2).reshape(2, 1, 1, 2) * small_noop
    #
    # Beginning Hadamard
    H = np.array([[1, 1], [1, -1]]) / np.sqrt(2.0)
    Hop = np.zeros((2, 2, 2, 2))
    Hop[1, 1, :, 1] = H[1, :]
    Hop[0, 0, :, 0] = H[0, :]
    #
    # Conditional rotations
    R0 = np.zeros((2, 2, 2, 2))
    R0[0, 0, 0, 0] = 1.0
    R0[0, 1, 1, 0] = 1.0
    R0[1, 0, 0, 1] = 1.0
    R1 = np.zeros((2, 2, 2, 2))
    R1[1, 1, 1, 1] = 1.0
    jϕ = sign * 1j * π

    # Place the Hadamard and rotations according to the instructions
    # in 'sites'. The first index is the control qubit, the other ones
    # are the following qubits in order of decreasing significance.
    def make_layer(sites):
        l = [noop] * N
        for i, ndx in enumerate(sites):
            if i == 0:
                l[ndx] = Hop
            else:
                l[ndx] = R0 + R1 * np.exp(jϕ / (2**i))
        for n, A in enumerate(l):
            if A is noop:
                l[n] = small_noop
            else:
                a, i, j, b = A.shape
                l[n] = np.sum(A, 0).reshape(1, i, j, b)
                break
        for n in reversed(range(N)):
            A = l[n]
            if A is noop:
                l[n] = small_noop
            else:
                a, i, j, b = A.shape
                l[n] = np.sum(A, -1).reshape(a, i, j, 1)
                break
        return MPO(l, **kwargs)

    return MPOList([make_layer(sites[i:]) for i in range(len(sites))], **kwargs)


def iqft_nd_mpo(sites: list[int], N: Optional[int] = None, **kwargs) -> MPOList:
    """Create an MPOList object representing the inverse Quantum Fourier Transform
    for subset of qubits in a quantum register with `N` qubits.

    Parameters
    ----------
    sites : list[int]
        List of qubits on which the transform acts
    N : int, default = len(sites)
        Number of qubits in the register.
    sign : int, default = -1
        Sign (+1 or -1) in the exponent of the transform. Defaults to
        the sign of the direct quantum Fourier transform.
    **kwargs :
        Other arguments accepted by :class:`MPO`

    Returns
    -------
    MPOList
        A sequence of :class:`MPO` that implements the transform.
    """
    return qft_nd_mpo(sites, N=N, sign=+1, **kwargs)
