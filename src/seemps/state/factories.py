from __future__ import annotations
from math import sqrt
import numpy as np
from collections.abc import Iterable
from ..typing import VectorLike, Tensor3
from .mps import MPS


def product_state(
    vectors: VectorLike | Iterable[VectorLike], length: int | None = None
) -> MPS:
    """Create a product state :class:`MPS`.

    Parameters
    ----------
    vectors : VectorLike | list[VectorLike]
        This may be a list of wavefunctions, or a single state vector that is
        repeated on all sites.
    length : int, optional
        If `vectors` is a single wavefunction, we need to know the size of the
        MPS, given as an integer value here.

    Returns
    -------
    MPS
        The quantum state in matrix-product state form.
    """

    def to_tensor(v: VectorLike | Iterable[VectorLike]) -> Tensor3:
        v = np.asarray(v)
        assert v.ndim == 1
        return v.reshape(1, v.size, 1)

    if length is not None:
        return MPS([to_tensor(vectors)] * length)  # type: ignore
    elif isinstance(vectors, Iterable):
        return MPS([to_tensor(v) for v in list(vectors)])
    raise Exception("Invalid argument to product_state")


def GHZ(n: int) -> MPS:
    """:class:`MPS` representing a GHZ state with `n` qubits."""
    a = np.zeros((2, 2, 2))
    b = a.copy()
    a[0, 0, 0] = a[0, 1, 1] = 1.0 / sqrt(2.0)
    b[0, 0, 0] = 1.0
    b[1, 1, 1] = 1.0
    data = [a] + [b] * (n - 1)
    data[0] = a[0:1, :, :]
    b = data[n - 1]
    data[n - 1] = b[:, :, 1:2] + b[:, :, 0:1]
    return MPS(data)


def W(n: int) -> MPS:
    """:class:`MPS` representing a W-state with `n` qubits.

    The W-state is defined as the quantum superpositions of all quantum states
    with a single spin up :math:`\\sum_i \\sigma_i^+ |000\\ldots\\rangle`
    """
    a = np.zeros((2, 2, 2))
    a[0, 0, 0] = 1.0
    a[0, 1, 1] = 1.0 / sqrt(n)
    a[1, 0, 1] = 1.0
    data = [a] * n
    data[0] = a[0:1, :, :]
    data[n - 1] = data[n - 1][:, :, 1:2]
    return MPS(data)


def spin_wave(state: VectorLike) -> MPS:
    """:class:`MPS` representing a spin-wave state with one excitation.

    The `state` is a wavepacket with `N` elements, representing the weights
    of the quantum excitation on each of the `N` qubits or spins. More
    precisely, `spin_wave(f)` represents
    :math:`\\sum_{i=0}^{N-1} f[i] \\sigma^+ |000\\ldots\\rangle`
    """
    ψ = np.array(state)
    data = [ψ] * ψ.size
    for n in range(0, ψ.size):
        B = np.zeros((2, 2, 2), dtype=ψ.dtype)
        B[0, 0, 0] = B[1, 0, 1] = 1.0
        B[0, 1, 1] = ψ[n]
        data[n] = B
    data[0] = data[0][0:1, :, :]
    data[-1] = data[-1][:, :, 1:]
    return MPS(data)


def graph_state(n: int) -> MPS:
    """Create an :class:`MPS` for a one-dimensional graph state with `n` qubits."""
    # Choose entangled pair state as : |00>+|11>
    # Apply Hadamard H on the left virtual spins
    # (which are the right spins of the entangled bond pairs)
    assert n > 1
    H = np.array([[1, 1], [1, -1]])
    # which gives |0>x(|0>+|1>)+|1>x(|0>-|1>) = |00>+|01>+|10>-|11>
    # Project as  |0><00| + |1><11|
    # We get the following MPS projectors:
    A0 = np.dot(np.array([[1, 0], [0, 0]]), H)
    A1 = np.dot(np.array([[0, 0], [0, 1]]), H)
    AA = np.array([A0, A1])
    AA = np.swapaxes(AA, 0, 1)
    data = [AA] * n
    data[0] = np.dot(np.array([[[1, 0], [0, 1]]]), H)
    data[-1] = np.swapaxes(np.array([[[1, 0], [0, 1]]]), 0, 2) / sqrt(2**n)
    return MPS(data)


def AKLT(n: int) -> MPS:
    """Create an :class:`MPS` for the AKLT spin-1 state with `n` sites."""
    assert n > 1
    # Choose entangled pair state as : |00>+|11>
    # Apply i * Pauli Y matrix on the left virtual spins (which are the right spins of the entangled bond pairs)
    iY = np.array([[0, 1], [-1, 0]])
    # which gives -|01>+|10>
    # Project as  |-1><00| +|0> (<01|+ <10|)/ \sqrt(2)+ |1><11|
    # We get the following MPS projectors:
    A0 = np.dot(np.array([[1, 0], [0, 0]]), iY)
    A1 = np.dot(np.array([[0, 1], [1, 0]]), iY)
    A2 = np.dot(np.array([[0, 0], [0, 1]]), iY)

    AA = np.array([A0, A1, A2]) / sqrt(2)
    AA = np.swapaxes(AA, 0, 1)
    data = [AA] * n
    data[-1] = np.array([[[1, 0], [0, 1], [0, 0]]])
    data[0] = np.array(np.einsum("ijk,kl->ijl", data[-1], iY)) / sqrt(2)
    data[-1] = np.swapaxes(data[-1], 0, 2)

    return MPS(data)


def random_uniform_mps(
    d: int,
    N: int,
    D: int = 1,
    truncate: bool = True,
    complex: bool = False,
    rng: np.random.Generator | None = None,
) -> MPS:
    """Create a random state with `N` elements of dimension `d` and bond
    dimension `D`.

    Parameters
    ----------
    d : int
        The dimension of each quantum system
    N : int
        The number of quantum systems in this :class:`MPS`
    D : int, default = 1
        The maximum bond dimension
    truncate : bool, default = True
        Do not reach `D` for tensors that do not require it.
    complex : bool, default = False
        If true, return states with complex wavefunctions.
    rng : np.random.Generator, default = np.random.default_rng()
        Random number generator used to create the state. Provide a seeded
        generator to ensure reproducibility

    Returns
    -------
    MPS
        A random matrix-product state.
    """
    return random_mps([d] * N, D, truncate, complex, rng)


def random_mps(
    dimensions: list[int],
    D: int = 1,
    truncate: bool = True,
    complex: bool = False,
    rng: np.random.Generator | None = None,
) -> MPS:
    """Create a random state with `N` elements of dimension `d` and bond
    dimension `D`.

    Parameters
    ----------
    dimensions : list[int]
        List of physical dimensions of each quantum system
    D : int, default = 1
        The maximum bond dimension
    truncate : bool, default = True
        Do not reach `D` for tensors that do not require it.
    complex : bool, default = False
        If true, return states with complex wavefunctions.
    rng : np.random.Generator, default = np.random.default_rng()
        Random number generator used to create the state. Provide a seeded
        generator to ensure reproducibility

    Returns
    -------
    MPS
        A random matrix-product state.
    """
    N = len(dimensions)
    mps: list[np.ndarray] = [np.ndarray(())] * N
    if rng is None:
        rng = np.random.default_rng()
    DR = 1
    if N > 60:
        truncate = False
    for i, d in enumerate(dimensions):
        DL = DR
        if not truncate and i != N - 1:
            DR = D
        else:
            DR = np.min([DR * d, D, d ** (N - i - 1)])
        T = rng.normal(size=(DL, d, DR))
        if complex:
            mps[i] = T + 1j * rng.normal(size=T.shape)
        else:
            mps[i] = T
    return MPS(mps)


def gaussian(n: int, x0: float, w0: float, k0: float) -> MPS:
    #
    # Return a W state with `n` components in MPS form or
    # in vector form
    #
    xx = np.arange(n)
    coefs = np.exp(-((xx - x0) ** 2) / w0**2 + 1j * k0 * xx)
    return spin_wave(coefs / np.linalg.norm(coefs))
