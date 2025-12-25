from __future__ import annotations
import math
import warnings
import numpy as np
from math import sqrt
from collections.abc import Sequence, Iterable
from ..tools import InvalidOperation
from ..typing import (
    Environment,
    Weight,
    Vector,
    VectorLike,
    to_dense_operator,
    Operator,
    Tensor3,
)
from . import array
from ..cython.core import DEFAULT_STRATEGY, Strategy
from .schmidt import _vector2mps


class MPS(array.TensorArray):
    """MPS (Matrix Product State) class.

    This implements a bare-bones Matrix Product State object with open
    boundary conditions. The tensors have three indices, `A[α,d,β]`, where
    `α,β` are the internal labels and `d` is the physical state of the given
    site.

    Parameters
    ----------
    data : Iterable[Tensor3]
        Sequence of three-legged tensors `A[α,si,β]`. The dimensions are not
        verified for consistency.
    error : float, default=0.0
        Accumulated truncation error in the previous tensors.
    """

    _error: float

    #
    # This class contains all the matrices and vectors that form
    # a Matrix-Product State.
    #
    __array_priority__: int = 10000

    def __init__(
        self,
        data: Iterable[np.ndarray],
        error: float = 0,
    ):
        super().__init__(data)
        self._error = error

    def copy(self) -> MPS:
        """Return a shallow copy of the MPS, without duplicating the tensors."""
        # We use the fact that TensorArray duplicates the list
        return MPS(self._data, self._error)

    def as_mps(self) -> MPS:
        return self

    def dimension(self) -> int:
        """Hilbert space dimension of this quantum system."""
        return math.prod(self.physical_dimensions())

    def physical_dimensions(self) -> list[int]:
        """List of physical dimensions for the quantum subsystems."""
        return list(a.shape[1] for a in self._data)

    def bond_dimensions(self) -> list[int]:
        """List of bond dimensions for the matrix product state.

        Returns a list or vector of `N+1` integers, for an MPS of size `N`.
        The integers `1` to `N-1` are the bond dimensions between the respective
        pairs of tensors. The first and last index are `1`, as it corresponds
        to a matrix product state with open boundary conditions.

        Returns
        -------
        list[int]
            List of virtual bond dimensions between MPS tensors, including the
            boundary ones.

        Examples
        --------
        >>> A = np.ones(1,2,3)
        >>> B = np.ones(3,2,1)
        >>> mps = MPS([A, B])
        >>> mps.bond_dimensions()
        [1, 3, 1]
        """
        return list(a.shape[0] for a in self._data) + [self._data[-1].shape[-1]]

    def max_bond_dimension(self) -> int:
        """Return the largest bond dimension."""
        return max(a.shape[-1] for a in self._data)

    def to_vector(self) -> Vector:
        """Convert this MPS to a state vector."""
        return _mps2vector(self._data)

    @classmethod
    def from_vector(
        cls,
        ψ: VectorLike,
        dimensions: Sequence[int],
        strategy: Strategy = DEFAULT_STRATEGY,
        normalize: bool = True,
        center: int = -1,
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
        center : int, default = -1
            Center of the canonicalized matrix product state

        Returns
        -------
        MPS
            A valid matrix-product state approximating this state vector.
        """
        data, error = _vector2mps(ψ, dimensions, strategy, normalize, center)
        return MPS(data, error)

    @classmethod
    def from_tensor(
        cls,
        state: np.ndarray,
        strategy: Strategy = DEFAULT_STRATEGY,
        normalize: bool = True,
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

    def __add__(self, state: MPS | MPSSum) -> MPSSum:
        """Represent `self + state` as :class:`.MPSSum`."""
        match state:
            case MPS():
                return MPSSum([1.0, 1.0], [self, state], check_args=False)
            case MPSSum(weights=w, states=s):
                return MPSSum([1.0] + w, [self] + s, check_args=False)
            case _:
                raise InvalidOperation("+", self, state)

    def __sub__(self, state: MPS | MPSSum) -> MPSSum:
        """Represent `self - state` as :class:`.MPSSum`"""
        match state:
            case MPS():
                return MPSSum([1, -1], [self, state], check_args=False)
            case MPSSum(weights=w, states=s):
                return MPSSum([1] + [-wi for wi in w], [self] + s, check_args=False)
            case _:
                raise InvalidOperation("-", self, state)

    def __mul__(self, n: Weight | MPS) -> MPS:
        """Compute `n * self` where `n` is a scalar."""
        match n:
            case int() | float() | complex():
                if n:
                    mps_mult = self.copy()
                    mps_mult._data[0] = n * mps_mult._data[0]
                    mps_mult._error *= abs(n)
                    return mps_mult
                return self.zero_state()
            case MPS():
                return MPS(
                    [
                        # np.einsum('aib,cid->acibd', A, B)
                        (
                            A[:, np.newaxis, :, :, np.newaxis] * B[:, :, np.newaxis, :]
                        ).reshape(A.shape[0] * B.shape[0], A.shape[1], -1)
                        for A, B in zip(self._data, n._data)
                    ]
                )
            case _:
                raise InvalidOperation("*", self, n)

    def __rmul__(self, n: Weight) -> MPS:
        """Compute `self * n`, where `n` is a scalar."""
        match n:
            case int() | float() | complex():
                if n:
                    mps_mult = self.copy()
                    mps_mult._data[0] = n * mps_mult._data[0]
                    mps_mult._error *= abs(n)
                    return mps_mult
                return self.zero_state()
            case _:
                raise InvalidOperation("*", n, self)

    def norm2(self) -> float:
        """Deprecated alias for :py:meth:`norm_squared`."""
        warnings.warn(
            "method norm2 is deprecated, use norm_squared", category=DeprecationWarning
        )
        return self.norm_squared()

    def norm_squared(self) -> float:
        """Norm-2 squared :math:`\\Vert{\\psi}\\Vert^2` of this MPS."""
        return abs(scprod(self, self).real)

    def norm(self) -> float:
        """Norm-2 :math:`\\Vert{\\psi}\\Vert^2` of this MPS."""
        return sqrt(abs(scprod(self, self)))

    def zero_state(self) -> MPS:
        """Return a zero wavefunction with the same physical dimensions."""
        return MPS([np.zeros((1, A.shape[1], 1)) for A in self._data])

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
        OL = _update_left_environment(A, np.matmul(to_dense_operator(O), A), ρL)
        ρR = self.right_environment(site)
        return _join_environments(OL, ρR)

    def expectation2(
        self, Opi: Operator, Opj: Operator, i: int, j: int | None = None
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
        Opi = to_dense_operator(Opi)
        Opj = to_dense_operator(Opj)
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

    def all_expectation1(self, operator: Operator | list[Operator]) -> Vector:
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
            op_i = operator[i] if isinstance(operator, list) else operator
            OρL = _update_left_environment(A, np.matmul(to_dense_operator(op_i), A), ρL)
            output[i] = _join_environments(OρL, ρR)
            ρL = _update_left_environment(A, A, ρL)
        return np.array(output)

    def left_environment(self, site: int) -> Environment:
        """Environment matrix for systems to the left of `site`."""
        ρ = _begin_environment()
        for A in self._data[:site]:
            ρ = _update_left_environment(A, A, ρ)
        return ρ

    def right_environment(self, site: int) -> Environment:
        """Environment matrix for systems to the right of `site`."""
        ρ = _begin_environment()
        for A in self._data[-1:site:-1]:
            ρ = _update_right_environment(A, A, ρ)
        return ρ

    def error(self) -> float:
        """Upper bound of the accumulated truncation error on this state.

        If this quantum state results from `N` steps in which we have obtained
        truncation errors :math:`\\delta_i`, this function returns the estimate
        :math:`\\sum_{i}\\delta_i`.

        Returns
        -------
        float
            Upper bound for the actual error when approximating this state.
        """
        return self._error

    def update_error(self, delta: float) -> None:
        """Register an increase in the truncation error.

        Parameters
        ----------
        delta : float
            Error increment in norm-2

        See also
        --------
        :py:meth:`error` : Total accumulated error after this update.
        """
        self._error += delta

    # TODO: We have to change the signature and working of this function, so that
    # 'sites' only contains the locations of the _new_ sites, and 'L' is no longer
    # needed. In this case, 'dimensions' will only list the dimensions of the added
    # sites, not all of them.
    def extend(
        self,
        L: int,
        sites: Sequence[int] | None = None,
        dimensions: int | list[int] = 2,
        state: Vector | None = None,
    ):
        """Enlarge an MPS so that it lives in a Hilbert space with `L` sites.

        Parameters
        ----------
        L : int
            The new size of the MPS. Must be strictly larger than `self.size`.
        sites : Iterable[int], optional
            Sequence of integers describing the sites that occupied by the
            tensors in this state.
        dimensions : int | list[int], default = 2
            Dimension of the added sites. It can be the same integer or a list
            of integers with the same length as `sites`.

        Returns
        -------
        MPS
            The extended MPS.

        Examples
        --------
        >>> import seemps.state
        >>> mps = seemps.state.random_uniform_mps(2, 10)
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

    def conj(self) -> MPS:
        """Return the complex-conjugate of this quantum state."""
        output = self.copy()
        for i, A in enumerate(output._data):
            output._data[i] = A.conj()
        return output

    def reverse(self) -> MPS:
        """Reverse the sites and tensors.

        Creates a new matrix product operator where tensors `0, 1, ..., N-1`
        are mapped to `N-1, N-2, ..., 0`. For the MPS to be consistent, this
        also implies reversing the order of the intermediate indices. Thus,
        if we label as `A` and `B` the tensors of the original and of the
        reversed MPOs, we have

        .. math::
            B_{a_{n-1},i_n,a_n} = A_{a_{N-n-1},i_{N-n-1},a_{N-n-2}}
        """
        return MPS(
            [np.moveaxis(op, [0, 1, 2], [2, 1, 0]) for op in reversed(self._data)],
        )


def _mps2vector(data: list[Tensor3]) -> Vector:
    #
    # Input:
    #  - data: list of tensors for the MPS (unchecked)
    # Output:
    #  - Ψ: Vector of complex Complexs with all the wavefunction amplitudes
    #
    # We keep Ψ[D,β], a tensor with all matrices contracted so far, where
    # 'D' is the dimension of the physical subsystems up to this point and
    # 'β' is the last uncontracted internal index.
    #
    Ψ: np.ndarray = np.ones(1)
    for A in reversed(data):
        α, d, β = A.shape
        # Ψ = np.einsum("Da,akb->Dkb", Ψ, A)
        Ψ = np.dot(A.reshape(α * d, β), Ψ).reshape(α, -1)
    return Ψ.reshape(-1)


from .mpssum import MPSSum  # noqa: E402
from .environments import (  # noqa: E402
    _begin_environment,
    _update_left_environment,
    _update_right_environment,
    _join_environments,
    scprod,
)
