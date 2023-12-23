def _mpo_multiply_tensor(A, B):
    # Implements
    # np.einsum("cjd,aijb->caidb", B, A)
    #
    # Matmul takes two arguments
    #     B(c, 1, 1, d, j)
    #     A(1, a, i, j, b)
    # It broadcasts, repeating the indices that are of size 1
    #     B(c, a, i, d, j)
    #     A(c, a, i, j, b)
    # And then multiplies the matrices that are formed by the last two
    # indices, (d,j) * (j,b) -> (b,d) so that the outcome has size
    #     C(c, a, i, d, b)
    #
    a, i, j, b = A.shape
    c, j, d = B.shape
    # np.matmul(...) -> C(a,i,b,c,d)
    return np.matmul(
        B.transpose(0, 2, 1).reshape(c, 1, 1, d, j), A.reshape(1, a, i, j, b)
    ).reshape(c * a, i, d * b)

cdef _mpo_apply_tensors(Alist, MPS mps):
    cdef:
        Py_ssize_t k, L = cpython.PyList_GET_SIZE(Alist)
        list Blist, output
    if L != mps._size:
        raise Exception("Mismatch in MPO and MPS size in '*' operator")
    Blist = mps._data
    output = [_mpo_multiply_tensor(A, B) for (A, B) in zip(Alist, Blist)]
    return _MPS_from_data(output, L, mps._error)

cdef MPO _MPO_from_data(list data, Py_ssize_t size, Strategy strategy):
    cdef MPO output = MPO.__new__(MPO)
    output._data = data
    output._size = size
    output._strategy = strategy
    return output

cdef class MPO(TensorArray):
    """Matrix Product Operator class.

    This implements a bare-bones Matrix Product Operator object with open
    boundary conditions. The tensors have four indices, A[α,i,j,β], where
    'α,β' are the internal labels and 'i,j' the physical indices ar the given
    site.

    Parameters
    ----------
    data: list[Tensor4]
        List of four-legged tensors forming the structure.
    strategy: Strategy, default = DEFAULT_STRATEGY
        Truncation strategy for algorithms.
    """

    __array_priority__ = 10000

    def __init__(self, data: list[Tensor4], strategy: Strategy = DEFAULT_STRATEGY):
        super().__init__(data)
        self._strategy = strategy

    def copy(self) -> MPO:
        """Return a shallow copy of the MPO, without duplicating the tensors."""
        # We use the fact that TensorArray duplicates the list
        return _MPO_from_data(list(self._data), self._size, self._strategy)

    def __add__(self, A: Union[MPO, MPOList, MPOSum]) -> MPOSum:
        """Represent `self + A` as :class:`.MPOSum`."""
        if isinstance(A, (MPO, MPOList)):
            return MPOSum([self, A], [1.0, 1.0])
        if isinstance(A, MPOSum):
            return MPOSum([self] + A.mpos, [1.0] + A.weights, A._strategy)
        raise TypeError(f"Cannod add MPO and {type(A)}")

    def __sub__(self, A: Union[MPO, MPOList, MPOSum]) -> MPOSum:
        """Represent `self - A` as :class:`.MPOSum`."""
        if isinstance(A, (MPO, MPOList)):
            return MPOSum([self, A], [1.0, -1.0])
        if isinstance(A, MPOSum):
            return MPOSum([self] + A.mpos, [1.0] + [-w for w in A.weights], A._strategy)
        raise TypeError(f"Cannod subtract MPO and {type(A)}")

    # TODO: The deep copy also copies the tensors. This should be improved.
    def __mul__(self, n: Weight) -> MPO:
        """Multiply an MPO by a scalar `n * self`"""
        if isinstance(n, (int, float, complex)):
            mpo_mult = self.copy()
            mpo_mult._data[0] = n * mpo_mult._data[0]
            return mpo_mult
        raise InvalidOperation("*", self, n)

    def __rmul__(self, n: Weight) -> MPO:
        """Multiply an MPO by a scalar `self * self`"""
        if isinstance(n, (int, float, complex)):
            mpo_mult = self.copy()
            mpo_mult._data[0] = n * mpo_mult._data[0]
            return mpo_mult
        raise InvalidOperation("*", n, self)

    # TODO: Rename to physical_dimensions()
    def dimensions(self) -> list[int]:
        """Return the physical dimensions of the MPO."""
        cdef:
            list dims = cpython.PyList_New(self._size)
            cnp.ndarray t
            Py_ssize_t k
        for k in range(self._size):
            t = <object>cpython.PyList_GET_ITEM(self._data, k)
            cpython.PyList_SetItem(dims, k, t.shape[1])
        return dims

    def bond_dimensions(self) -> list[int]:
        """Return the bond dimensions of the MPO."""
        cdef:
            list dims = cpython.PyList_New(self._size+1)
            cnp.ndarray t
            Py_ssize_t k
        for k in range(self._size):
            t = <object>cpython.PyList_GET_ITEM(self._data, k)
            cpython.PyList_SetItem(dims, k, t.shape[0])
        cpython.PyList_SetItem(dims, self._size, 1)
        return dims

    # TODO: Rename to to_matrix()
    def tomatrix(self) -> Operator:
        """Convert this MPO to a dense or sparse matrix."""
        D = 1  # Total physical dimension so far
        out = np.array([[[1.0]]])
        for A in self._data:
            _, i, _, b = A.shape
            out = np.einsum("lma,aijb->limjb", out, A)
            D *= i
            out = out.reshape(D, D, b)
        return out[:, :, 0]

    def set_strategy(self, strategy):
        """Return MPO with the given strategy."""
        # TODO: We do not need to create copies and in any case
        # the list should be copied!
        return _MPO_from_data(self._data, self._size, self._strategy)

    def apply(
        self,
        state: Union[MPS, MPSSum],
        strategy: Optional[Strategy] = None,
    ) -> Union[MPS, MPSSum]:
        """Implement multiplication `A @ state` between a matrix-product operator
        `A` and a matrix-product state `state`.

        Parameters
        ----------
        state : MPS | MPSSum
            Transformed state.
        strategy : Strategy, optional
            Truncation strategy, defaults to DEFAULT_STRATEGY

        Returns
        -------
        CanonicalMPS
            The result of the contraction.
        """
        cdef Strategy the_strategy

        if isinstance(state, MPS):
            state = _mpo_apply_tensors(self._data, <MPS>state)
        elif isinstance(state, MPSSum):
            state = _MPSSum_from_data(
                (<MPSSum>state)._weights,
                [_mpo_apply_tensors(self._data, mps)
                 for mps in (<MPSSum>state)._states],
                (<MPSSum>state)._size
            )
        else:
            raise TypeError(f"Cannot multiply MPO with {state}")

        the_strategy = self._strategy if strategy is None else strategy
        if the_strategy.simplify != SIMPLIFICATION_DO_NOT_SIMPLIFY:
            state = truncate.simplify(state, strategy=the_strategy)
        return state

    def __matmul__(self, b: Union[MPS, MPSSum]) -> Union[MPS, MPSSum]:
        """Implement multiplication `self @ b`."""
        return self.apply(b)

    # TODO: We have to change the signature and working of this function, so that
    # 'sites' only contains the locations of the _new_ sites, and 'L' is no longer
    # needed. In this case, 'dimensions' will only list the dimensions of the added
    # sites, not all of them.
    def extend(
        self,
        L: int,
        sites: Optional[Sequence[int]] = None,
        dimensions: Union[int, list[int]] = 2,
    ) -> MPO:
        """Enlarge an MPO so that it acts on a larger Hilbert space with 'L' sites.

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
        MPO
            Extended MPO.
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
                d = final_dimensions[k]
                A = np.eye(D).reshape(D, 1, 1, D) * np.eye(d).reshape(1, d, d, 1)
                data[i] = A
                k = k + 1
            else:
                D = A.shape[3]
        return MPO(data, self._strategy)

    def expectation(self, bra: MPS, ket: Optional[MPS] = None) -> Weight:
        """Expectation value of MPO on one or two MPS states.

        If one state is given, this state is interpreted as :math:`\\psi`
        and this function computes :math:`\\langle{\\psi|O\\psi}\\rangle`
        If two states are given, the first one is the bra :math:`\\psi`,
        the second one is the ket :math:`\\phi`, and this computes
        :math:`\\langle\\psi|O|\\phi\\rangle`.

        Parameters
        ----------
        bra : MPS
            The state :math:`\\psi` on which the expectation value
            is computed.
        ket : Optional[MPS]
            The ket component of the expectation value. Defaults to `bra`.

        Returns
        -------
        float | complex
            :math:`\\langle\\psi\\vert{O}\\vert\\phi\\rangle` where `O`
            is the matrix-product operator.
        """
        if isinstance(bra, CanonicalMPS):
            center = bra.center
        elif isinstance(bra, MPS):
            center = self.size - 1
        else:
            raise Exception("MPS required")
        if ket is None:
            ket = bra
        elif not isinstance(ket, MPS):
            raise Exception("MPS required")
        left = right = begin_mpo_environment()
        operators = self._data
        for i in range(0, center):
            left = update_left_mpo_environment(
                left, bra[i].conj(), operators[i], ket[i]
            )
        for i in range(self.size - 1, center - 1, -1):
            right = update_right_mpo_environment(
                right, bra[i].conj(), operators[i], ket[i]
            )
        return join_mpo_environments(left, right)

    @property
    def strategy(self) -> Strategy:
        return self._strategy
