cdef MPSSum _MPSSum_from_data(list weights, list states, Py_ssize_t size):
    cdef MPSSum output = MPSSum.__new__(MPSSum)
    output._weights = weights
    output._states = states
    output._size = size
    return output


cdef class MPSSum:
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
    """
    #
    # This class contains all the matrices and vectors that form
    # a Matrix-Product State.
    #
    __array_priority__ = 10000

    def __init__(
        self,
        weights: Sequence[Weight],
        states: Sequence[MPS],
    ):
        # TODO: This is not consistent with MPS, MPO and MPOSum
        # which copy their input lists. We should decide whether we
        # want to copy or not.
        assert len(states) == len(weights)
        assert len(states) > 0
        self._weights = list(weights)
        self._states = list(states)
        for A in self._states:
            if not isinstance(A, MPS):
                raise Exception("MPSSum argument error")
        cdef MPS one_state = states[0]
        self._size = one_state._size

    def copy(self) -> MPSSum:
        """Return a shallow copy of the MPS sum and its data. Does not copy
        the states, only the list that stores them."""
        return _MPSSum_from_data(list(self._weights), list(self._states), self._size)

    def __add__(self, state: Union[MPS, MPSSum]) -> MPSSum:
        """Add `self + state`, incorporating it to the lists."""
        cdef MPSSum sumstate
        if isinstance(state, MPS):
            return _MPSSum_from_data(
                self._weights + [1.0],
                self._states + [state],
                self._size
            )
        elif isinstance(state, MPSSum):
            sumstate = state
            return _MPSSum_from_data(
                self._weights + sumstate._weights,
                self._states + sumstate._states,
                self._size
            )
        raise InvalidOperation("+", self, state)

    def __sub__(self, state: Union[MPS, MPSSum]) -> MPSSum:
        """Subtract `self - state`, incorporating it to the lists."""
        cdef MPSSum sumstate
        if isinstance(state, MPS):
            return _MPSSum_from_data(
                self._weights + [-1],
                self._states + [state],
                self._size
            )
        if isinstance(state, MPSSum):
            sumstate = state
            return _MPSSum_from_data(
                self._weights + [-w for w in sumstate._weights],
                self._states + sumstate._states,
                self._size
            )
        raise InvalidOperation("-", self, state)

    def __mul__(self, n: Weight) -> MPSSum:
        """Rescale the linear combination `n * self` for scalar `n`."""
        if isinstance(n, (int, float, complex)):
            return _MPSSum_from_data(
                [n * w for w in self._weights],
                self._states,
                self._size
            )
        raise InvalidOperation("*", self, n)

    def __rmul__(self, n: Weight) -> MPSSum:
        """Rescale the linear combination `self * n` for scalar `n`."""
        if isinstance(n, (int, float, complex)):
            return _MPSSum_from_data(
                [n * w for w in self._weights],
                self._states,
                self._size
            )
        raise InvalidOperation("*", n, self)

    def to_vector(self) -> Vector:
        """Return the wavefunction of this quantum state."""
        return sum(wa * A.to_vector() for wa, A in zip(self._weights, self._states))  # type: ignore

    def _joined_tensors(self, i: int, L: int) -> Tensor3:
        """Join the tensors from all MPS into bigger tensors."""
        As: list[Tensor3] = [s[i] for s in self._states]
        if i == 0:
            return np.concatenate([w * A for w, A in zip(self._weights, As)], axis=2)
        if i == L - 1:
            return np.concatenate(As, axis=0)

        DL: int = 0
        DR: int = 0
        d: int
        w: Weight = 0
        for A in As:
            a, d, b = A.shape
            DL += a
            DR += b
            w += A[0, 0, 0]
        B = np.zeros((DL, d, DR), dtype=type(w))
        DL = 0
        DR = 0
        for A in As:
            a, d, b = A.shape
            B[DL : DL + a, :, DR : DR + b] = A
            DL += a
            DR += b
        return B

    def join(
        self,
        canonical: bool = True,
        center: Optional[int] = None,
        strategy: Strategy = DEFAULT_STRATEGY,
    ):
        """Create an `MPS` or `CanonicalMPS` state by combining all tensors
        from all states in the linear combination.

        Parameters
        ----------
        canonical: bool
            Whether to create the state in canonical form. Defaults to `True`.
        center: Optional[int]
            Center for the `CanonicalMPS`, if `canonical` is true.
        strategy: Strategy, default = DEFAULT_STRATEGY
            Parameters for the truncation algorithms used when creating the
            `CanonicalMPS`. Only used if `canonical` is `True`.

        Returns
        -------
        MPS | CanonicalMPS
            Quantum state approximating this sum.
        """
        L = self._size
        data = [self._joined_tensors(i, L) for i in range(L)]
        if canonical:
            return CanonicalMPS(
                data,
                center=center,
                strategy=strategy,
            )
        else:
            return MPS(data)

    def conj(self) -> MPSSum:
        """Return the complex-conjugate of this quantum state."""
        return _MPSSum_from_data(
            [np.conj(w) for w in self._weights],
            [state.conj() for state in self._states],
            self._size
        )

    @property
    def states(self):
        return self._states

    @property
    def weights(self):
        return self._weights

    @property
    def size(self):
        return self._size
