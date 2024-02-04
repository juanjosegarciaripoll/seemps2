from __future__ import annotations
import math
import warnings
import numpy as np
from ..tools import InvalidOperation
from ..typing import *
from . import array
from .core import DEFAULT_STRATEGY, Strategy
from .schmidt import vector2mps


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
    __array_priority__ = 10000

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
        return MPS(vector2mps(ψ, dimensions, strategy, normalize))

    @classmethod
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

    def __add__(self, state: Union[MPS, MPSSum]) -> MPSSum:
        """Represent `self + state` as :class:`.MPSSum`."""
        if isinstance(state, MPS):
            return MPSSum([1.0, 1.0], [self, state])
        if isinstance(state, MPSSum):
            return MPSSum([1.0] + state.weights, [self] + state.states)
        raise InvalidOperation("+", self, state)

    def __sub__(self, state: Union[MPS, MPSSum]) -> MPSSum:
        """Represent `self - state` as :class:`.MPSSum`"""
        if isinstance(state, MPS):
            return MPSSum([1, -1], [self, state])
        if isinstance(state, MPSSum):
            return MPSSum(
                [1] + list((-1) * np.asarray(state.weights)),
                [self] + state.states,
            )
        raise InvalidOperation("-", self, state)

    def __mul__(self, n: Union[Weight, MPS]) -> MPS:
        """Compute `n * self` where `n` is a scalar."""
        if isinstance(n, (int, float, complex)):
            mps_mult = self.copy()
            mps_mult._data[0] = n * mps_mult._data[0]
            mps_mult._error = np.abs(n) ** 2 * mps_mult._error
            return mps_mult
        if isinstance(n, MPS):
            return self.wavefunction_product(n)
        raise InvalidOperation("*", self, n)

    def __rmul__(self, n: Weight) -> MPS:
        """Compute `self * n`, where `n` is a scalar."""
        if isinstance(n, (int, float, complex)):
            mps_mult = self.copy()
            mps_mult._data[0] = n * mps_mult._data[0]
            mps_mult._error = np.abs(n) ** 2 * mps_mult._error
            return mps_mult
        raise InvalidOperation("*", n, self)

    def norm2(self) -> float:
        """Deprecated alias for :py:meth:`norm_squared`."""
        warnings.warn(
            "method norm2 is deprecated, use norm_squared", category=DeprecationWarning
        )
        return self.norm_squared()

    def norm_squared(self) -> float:
        """Norm-2 squared :math:`\\Vert{\\psi}\\Vert^2` of this MPS."""
        return abs(scprod(self, self))

    def norm(self) -> float:
        """Norm-2 :math:`\\Vert{\\psi}\\Vert^2` of this MPS."""
        return np.sqrt(abs(scprod(self, self)))

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
        OL = update_left_environment(A, A, ρL, operator=O)
        ρR = self.right_environment(site)
        return join_environments(OL, ρR)

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
                OQL = update_left_environment(A, A, OQL, operator=Opi)
            elif ndx == j:
                OQL = update_left_environment(A, A, OQL, operator=Opj)
            else:
                OQL = update_left_environment(A, A, OQL)
        return join_environments(OQL, self.right_environment(j))

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
        ρ = begin_environment()
        allρR: list[Environment] = [ρ] * L
        for i in range(L - 1, 0, -1):
            A = self[i]
            ρ = update_right_environment(A, A, ρ)
            allρR[i - 1] = ρ

        ρL = begin_environment()
        output: list[Weight] = [0.0] * L
        for i in range(L):
            A = self[i]
            ρR = allρR[i]
            OρL = update_left_environment(
                A,
                A,
                ρL,
                operator=operator[i] if isinstance(operator, list) else operator,
            )
            output[i] = join_environments(OρL, ρR)
            ρL = update_left_environment(A, A, ρL)
        return np.array(output)

    def left_environment(self, site: int) -> Environment:
        """Environment matrix for systems to the left of `site`."""
        ρ = begin_environment()
        for A in self._data[:site]:
            ρ = update_left_environment(A, A, ρ)
        return ρ

    def right_environment(self, site: int) -> Environment:
        """Environment matrix for systems to the right of `site`."""
        ρ = begin_environment()
        for A in self._data[-1:site:-1]:
            ρ = update_right_environment(A, A, ρ)
        return ρ

    def error(self) -> float:
        """Upper bound of the accumulated truncation error on this state.

        If this quantum state results from `N` steps in which we have obtained
        truncation errors :math:`\\delta_i`, this function returns the estimate
        :math:`\\sqrt{\\sum_{i}\\delta_i^2}`.

        Returns
        -------
        float
            Upper bound for the actual error when approximating this state.
        """
        return self._error

    def update_error(self, delta: float) -> float:
        """Register an increase in the truncation error.

        Parameters
        ----------
        delta : float
            Error increment in norm-2

        Returns
        -------
        float
            Accumulated upper bound of total truncation error.

        See also
        --------
        :py:meth:`error` : Total accumulated error after this update.
        """
        self._error = (np.sqrt(self._error) + np.sqrt(delta)) ** 2
        return self._error

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

    def wavefunction_product(self, other: MPS) -> MPS:
        """Elementwise product of the wavefunctions of two quantum states.

        Given two MPS `self` and `other, return another MPS whose
        wavefunction is the element-wise product of those. This
        product grows the bond dimensions, combining the tensors
        `A[a,i,b]` and `B[c,i,d]` of both states into the composite
        `C[a*c,i,b*d]`. Naturally, the physical dimensions of `self`
        and `other` must be identical.

        Parameters
        ----------
        other : MPS
            Another quantum state.

        Returns
        -------
        MPS
            The state that results from the `self .* other`
        """

        def combine(A, B):
            # Combine both tensors
            a, d, b = A.shape
            c, db, e = B.shape
            if d != db:
                raise Exception("Non matching MPS physical dimensions.")
            # np.einsum('adb,cde->acdbe', A, B)
            C = A.reshape(a, 1, d, b, 1) * B.reshape(1, c, d, 1, e)
            return C.reshape(a * c, d, b * e)

        return MPS([combine(A, B) for A, B in zip(self, other)])

    def conj(self) -> MPS:
        """Return the complex-conjugate of this quantum state."""
        output = self.copy()
        for i, A in enumerate(output._data):
            output._data[i] = A.conj()
        return output


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
from .environments import *  # noqa: E402
