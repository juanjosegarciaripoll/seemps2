from __future__ import annotations
import math
import warnings
import numpy as np
from math import sqrt
from typing import Optional, Union, Sequence, Iterable
from ..tools import InvalidOperation
from ..typing import Weight, Vector, VectorLike, Operator, Tensor3
from .core import DEFAULT_STRATEGY, Strategy, TensorArray, MPS, MPSSum, CanonicalMPS
from .schmidt import _vector2mps
from .environments import (  # noqa: E402
    Environment,
    _begin_environment,
    _update_left_environment,
    _update_right_environment,
    _join_environments,
    scprod,
)


def _mps2vector(data: MPS) -> Vector:
    """Convert this MPS to a state vector."""
    Ψ: np.ndarray = np.ones(1)
    for A in reversed(data):
        α, d, β = A.shape
        # Ψ = np.einsum("Da,akb->Dkb", Ψ, A)
        Ψ = np.dot(A.reshape(α * d, β), Ψ).reshape(α, -1)
    return Ψ.reshape(-1)


@classmethod  # type: ignore
def from_vector(
    cls,
    ψ: VectorLike,
    dimensions: Sequence[int],
    strategy: Strategy = DEFAULT_STRATEGY,
    normalize: bool = True,
    center: int = -1,
    **kwdargs,
) -> MPS:
    """Create a matrix-product state from a state vector.

    Parameters
    ----------
    ψ : VectorLike
        Real or complex vector of a wavefunction.
    dimensions : Sequence[int]
        Sequence of integers representing the dimensions of the
        quantum systems that form this state.
    strategy : Strategy, default = DEFAULT_STRATEGY
        Default truncation strategy for algorithms working on this state.
    normalize : bool, default = True
        Whether the state is normalized to compensate truncation errors.

    Returns
    -------
    MPS
        A valid matrix-product state approximating this state vector.
    """
    data, error = _vector2mps(ψ, dimensions, strategy, normalize, center)
    return MPS(data, error)


@classmethod  # type: ignore
def from_tensor(
    cls,
    state: np.ndarray,
    strategy: Strategy = DEFAULT_STRATEGY,
    normalize: bool = True,
    **kwdargs,
) -> MPS:
    """Create a matrix-product state from a tensor that represents a
    composite quantum system.

    The tensor `state` must have `N>=1` indices, each of them associated
    to an individual quantum system, in left-to-right order. This function
    decomposes the tensor into a contraction of `N` three-legged tensors
    as expected from an MPS.

    Parameters
    ----------
    state : np.ndarray
        Real or complex tensor with `N` legs.
    strategy : Strategy, default = DEFAULT_STRATEGY
        Default truncation strategy for algorithms working on this state.
    normalize : bool, default = True
        Whether the state is normalized to compensate truncation errors.

    Returns
    -------
    MPS
        A valid matrix-product state approximating this state vector.
    """
    return cls.from_vector(state.reshape(-1), state.shape, strategy, normalize)


# TODO: We have to change the signature and working of this function, so that
# 'sites' only contains the locations of the _new_ sites, and 'L' is no longer
# needed. In this case, 'dimensions' will only list the dimensions of the added
# sites, not all of them.
def extend(
    self,
    L: int,
    sites: Optional[Sequence[int]] = None,
    dimensions: Union[int, list[int]] = 2,
    state: Optional[Vector] = None,
):
    """Enlarge an MPS so that it lives in a Hilbert space with `L` sites.

    Parameters
    ----------
    L : int
        The new size of the MPS. Must be strictly larger than `self.size`.
    sites : Iterable[int], optional
        Sequence of integers describing the sites that occupied by the
        tensors in this state.
    dimensions : Union[int, list[int]], default = 2
        Dimension of the added sites. It can be the same integer or a list
        of integers with the same length as `sites`.

    Returns
    -------
    MPS
        The extended MPS.

    Examples
    --------
    >>> import seemps.state
    >>> mps = seemps.state.random(2, 10)
    >>> mps.physical_dimensions()
    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
    >>> mps = mps.extend(12, [0, 2, 4, 5, 6, 7, 8, 9, 10, 11], 3)
    >>> mps.physical_dimensions()
    [2, 3, 2, 3, 2, 2, 2, 2, 2, 2, 2, 2]
    """
    if isinstance(dimensions, int):
        final_dimensions = [dimensions] * max(L - self.size, 0)
    else:
        final_dimensions = dimensions.copy()
        assert len(dimensions) == L - self.size
    if sites is None:
        sites = range(self.size)
    assert L >= self.size
    assert len(sites) == self.size

    data: list[np.ndarray] = [np.ndarray(())] * L
    for ndx, A in zip(sites, self):
        data[ndx] = A
    D = 1
    k = 0
    for i, A in enumerate(data):
        if A.ndim == 0:
            A = np.zeros((D, final_dimensions[k], D))
            if state is not None:
                A = np.eye(D).reshape(D, 1, D) * np.reshape(state, (-1, 1))
            else:
                A[:, 0, :] = np.eye(D)
            data[i] = A
            k += 1
        else:
            D = A.shape[-1]
    return MPS(data)


def all_expectation1(self, operator: Union[Operator, list[Operator]]) -> Vector:
    """Vector of expectation values of the given operator acting on all
    possible sites of the MPS.

    Parameters
    ----------
    operator : Operator | list[Operator]
        If `operator` is an observable, it is applied on each possible site.
        If it is a list, the expectation value of `operator[i]` is computed
        on the i-th site.

    Returns
    -------
    Vector
        Numpy array of expectation values.
    """
    L = self.size
    ρ = _begin_environment()
    allρR: list[Environment] = [ρ] * L
    for i in range(L - 1, 0, -1):
        A = self[i]
        ρ = _update_right_environment(A, A, ρ)
        allρR[i - 1] = ρ

    ρL = _begin_environment()
    output: list[Weight] = [0.0] * L
    for i in range(L):
        A = self[i]
        ρR = allρR[i]
        opi = operator[i] if isinstance(operator, list) else operator
        OρL = _update_left_environment(A, np.matmul(opi, A), ρL)
        output[i] = _join_environments(OρL, ρR)
        ρL = _update_left_environment(A, A, ρL)
    return np.array(output)


def expectation1(self, O: Operator, site: int) -> Weight:
    """Compute the expectation value :math:`\\langle\\psi|O_i|\\psi\\rangle`
    of an operator O acting on the `i`-th site

    Parameters
    ----------
    state : MPS
        Quantum state :math:`\\psi` used to compute the expectation value.
    O : Operator
        Local observable acting onto the `i`-th subsystem
    i : int
        Index of site, in the range `[0, state.size)`

    Returns
    -------
    float | complex
        Expectation value.
    """
    ρL = self.left_environment(site)
    A = self[site]
    OL = _update_left_environment(A, np.matmul(O, A), ρL)
    ρR = self.right_environment(site)
    return _join_environments(OL, ρR)


def expectation2(
    self, Opi: Operator, Opj: Operator, i: int, j: Optional[int] = None
) -> Weight:
    """Compute the expectation value :math:`\\langle\\psi|O_i Q_j|\\psi\\rangle`
    of two operators `O` and `Q` acting on the `i`-th and `j`-th subsystems.

    Parameters
    ----------
    state : MPS
        Quantum state :math:`\\psi` used to compute the expectation value.
    O, Q : Operator
        Local observables
    i : int
    j : int, default=`i+1`
        Indices of sites, in the range `[0, state.size)`

    Returns
    -------
    float | complex
        Expectation value.
    """
    if j is None:
        j = i + 1
    elif j == i:
        return self.expectation1(Opi @ Opj, i)
    elif j < i:
        i, j = j, i
        Opi, Opj = Opj, Opi
    OQL = self.left_environment(i)
    for ndx in range(i, j + 1):
        A = self[ndx]
        if ndx == i:
            OQL = _update_left_environment(A, np.matmul(Opi, A), OQL)
        elif ndx == j:
            OQL = _update_left_environment(A, np.matmul(Opj, A), OQL)
        else:
            OQL = _update_left_environment(A, A, OQL)
    return _join_environments(OQL, self.right_environment(j))


def norm2(self) -> float:
    """Deprecated alias for :py:meth:`norm_squared`."""
    warnings.warn(
        "method norm2 is deprecated, use norm_squared", category=DeprecationWarning
    )
    return self.norm_squared()


# MPS.__array_priority__ = 10000
MPS.norm2 = norm2  # type: ignore
MPS.to_vector = _mps2vector  # type: ignore
MPS.from_tensor = from_tensor  # type: ignore
MPS.from_vector = from_vector  # type: ignore
MPS.extend = extend  # type: ignore
MPS.all_expectation1 = all_expectation1  # type: ignore
MPS.expectation1 = expectation1  # type: ignore
MPS.expectation2 = expectation2  # type: ignore

__all__ = ["MPS"]
