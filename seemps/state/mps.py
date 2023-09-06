from __future__ import annotations
import copy
import math
import numpy as np
from ..typing import *
from ..tools import InvalidOperation
from .environments import *
from .schmidt import vector2mps
from .core import DEFAULT_STRATEGY, Strategy
from . import array
import warnings


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
    strategy : Strategy, default=DEFAULT_STRATEGY
        Default truncation strategy when operating on this state
    """

    _error: float
    strategy: Strategy

    #
    # This class contains all the matrices and vectors that form
    # a Matrix-Product State.
    #
    __array_priority__ = 10000

    def __init__(
        self,
        data: Iterable[np.ndarray],
        error: float = 0,
        strategy: Strategy = DEFAULT_STRATEGY,
    ):
        super(MPS, self).__init__(data)
        self._error = error
        self.strategy = strategy

    def dimension(self) -> int:
        """Hilbert space dimension of this quantum system."""
        return math.prod(self.physical_dimensions())

    def physical_dimensions(self) -> list[int]:
        """List of physical dimensions for the quantum subsystems."""
        return list(a.shape[1] for a in self._data)

    def bond_dimensions(self) -> list[int]:
        """List of bond dimensions for the matrix product state."""
        return list(a.shape[0] for a in self._data[:-1]) + [self._data[-1].shape[-1]]

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

    def __add__(self, state: Union[MPS, MPSSum]) -> MPSSum:
        """Represent `self + state` as :class:`.MPSSum`."""
        if isinstance(state, MPS):
            return MPSSum([1.0, 1.0], [self, state], self.strategy)
        if isinstance(state, MPSSum):
            return MPSSum([1.0] + state.weights, [self] + state.states, self.strategy)
        raise InvalidOperation("+", self, state)

    def __sub__(self, state: Union[MPS, MPSSum]) -> MPSSum:
        """Represent `self - state` as :class:`.MPSSum`"""
        if isinstance(state, MPS):
            return MPSSum([1, -1], [self, state], self.strategy)
        if isinstance(state, MPSSum):
            return MPSSum(
                [1] + list((-1) * np.asarray(state.weights)),
                [self] + state.states,
                self.strategy,
            )
        raise InvalidOperation("-", self, state)

    def __mul__(self, n: Weight) -> MPS:
        """Compute `n * self` where `n` is a scalar."""
        if isinstance(n, (float, complex)):
            mps_mult = copy.deepcopy(self)
            mps_mult._data[0] = n * mps_mult._data[0]
            mps_mult._error = np.abs(n) ** 2 * mps_mult._error
            return mps_mult
        if isinstance(n, MPS):
            return self.wavefunction_product(n)
        raise InvalidOperation("*", self, n)

    def __rmul__(self, n: Weight) -> MPS:
        """Compute `self * n`, where `n` is a scalar."""
        if isinstance(n, (float, complex)):
            mps_mult = copy.deepcopy(self)
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
        """Norm-2 squared :math:`\\Vert{\psi}\\Vert^2` of this MPS."""
        return abs(scprod(self, self))

    def norm(self) -> float:
        """Norm-2 :math:`\\Vert{\psi}\\Vert^2` of this MPS."""
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
        sites: Optional[Iterable[int]] = None,
        dimensions: Union[int, list[int]] = 2,
    ):
        """Enlarge an MPS so that it lives in a Hilbert space with `L` sites.

        Parameters
        ----------
        L : int
            The new size of the MPS. Must be strictly larger than `self.size`.

        sites : Iterable[int], optional
            Sequence of integers describing the sites that are new in the
            range `[0,L)`. All other sites are filled in order with the content
            from this MPS.
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
        >>> mps = mps.extend(12, [1, 3], 3)
        >>> mps.physical_dimensions()
        [3, 2, 3, 2, 3, 3, 3, 3, 3, 3, 3, 3]
        """
        assert L >= self.size
        if isinstance(dimensions, int):
            final_dimensions = [dimensions] * L
        else:
            final_dimensions = dimensions.copy()
        if sites is None:
            sites = range(self.size)

        data: list[np.ndarray] = [np.ndarray(())] * L
        for ndx, A in zip(sites, self):
            data[ndx] = A
            final_dimensions[ndx] = A.shape[1]
        D = 1
        for i, A in enumerate(data):
            if A.ndim == 0:
                d = final_dimensions[i]
                A = np.zeros((D, d, D))
                A[:, 0, :] = np.eye(D)
                data[i] = A
            else:
                D = A.shape[-1]
        return MPS(data, strategy=self.strategy)

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


class MPSSum:
    """Class representing a weighted sum (or difference) of two or more :class:`MPS`.

    This class is an intermediate representation for the linear combination of
    MPS quantum states. Assume that :math:`\\psi, \\phi` and :math:`\\xi` are
    MPS and :math:`a, b, c` some real or complex numbers. The addition
    :math:`a \\psi - b \\phi + c \\xi` can be stored as
    `MPSSum([a, -b, c], [ψ, ϕ, ξ])`.


    Parameters
    ----------
    weights : list[Weight]
        Real or complex numbers representing the weights of the linear combination.
    states : list[MPS]
        List of matrix product states weighted.
    strategy : Strategy, default = DEFAULT_STRATEGY
        Truncation strategy for later operations, or when converting this sum
        to a real MPS.
    """

    weights: list[Weight]
    states: list[MPS]
    strategy: Strategy

    #
    # This class contains all the matrices and vectors that form
    # a Matrix-Product State.
    #
    __array_priority__ = 10000

    def __init__(
        self,
        weights: list[Weight],
        states: list[MPS],
        strategy: Strategy = DEFAULT_STRATEGY,
    ):
        self.weights = weights
        self.states = states
        self.strategy = strategy

    def __add__(self, state: Union[MPS, MPSSum]) -> MPSSum:
        """Add `self + state`, incorporating it to the lists."""
        if isinstance(state, MPS):
            return MPSSum(
                self.weights + [1.0],
                self.states + [state],
                self.strategy,
            )
        elif isinstance(state, MPSSum):
            return MPSSum(
                self.weights + state.weights,
                self.states + state.states,
                self.strategy,
            )
        raise InvalidOperation("+", self, state)

    def __sub__(self, state: Union[MPS, MPSSum]) -> MPSSum:
        """Subtract `self - state`, incorporating it to the lists."""
        if isinstance(state, MPS):
            return MPSSum(self.weights + [-1], self.states + [state], self.strategy)
        if isinstance(state, MPSSum):
            return MPSSum(
                self.weights + [-w for w in state.weights],
                self.states + state.states,
                self.strategy,
            )
        raise InvalidOperation("-", self, state)

    def __mul__(self, n: Weight) -> MPSSum:
        """Rescale the linear combination `n * self` for scalar `n`."""
        if isinstance(n, (float, complex)):
            return MPSSum([n * w for w in self.weights], self.states, self.strategy)
        raise InvalidOperation("*", self, n)

    def __rmul__(self, n: Weight) -> MPSSum:
        """Rescale the linear combination `self * n` for scalar `n`."""
        if isinstance(n, (float, complex)):
            return MPSSum([n * w for w in self.weights], self.states, self.strategy)
        raise InvalidOperation("*", n, self)

    def to_vector(self) -> Vector:
        """Return the wavefunction of this quantum state."""
        return sum(wa * A.to_vector() for wa, A in zip(self.weights, self.states))  # type: ignore

    # TODO: Rename toMPS -> to_MPS
    def toMPS(
        self, normalize: Optional[bool] = None, strategy: Optional[Strategy] = None
    ) -> MPS:
        """Approximate the linear combination with a new :class:`MPS`.

        This routine applies the :func:`~seemps.truncate.simplify` algorithm with
        the given truncation strategy, optionally normalizing the state. The
        result is a new :class:`MPS` with some approximation error.

        Parameters
        ----------
        normalize : bool, default = None
            Normalize the state after performing the approximation.
        strategy : Strategy, default = DEFAULT_STRATEGY
            Parameters for the simplificaiton and truncation algorithms

        Returns
        -------
        MPS
            Quantum state approximating this sum.
        """
        from ..truncate.combine import combine

        if strategy is None:
            strategy = self.strategy
        ψ, _ = combine(
            self.weights,
            self.states,
            maxsweeps=strategy.get_max_sweeps(),
            tolerance=strategy.get_tolerance(),
            normalize=strategy.get_normalize_flag() if normalize is None else normalize,
            max_bond_dimension=strategy.get_max_bond_dimension(),
        )
        return ψ

    def conj(self) -> MPSSum:
        """Return the complex-conjugate of this quantum state."""
        return MPSSum(
            [np.conj(w) for w in self.weights],
            [state.conj() for state in self.states],
            self.strategy,
        )
