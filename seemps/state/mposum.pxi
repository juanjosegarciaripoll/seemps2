cdef class MPOSum(object):
    """Object representing a linear combination of matrix-product opeators.

    Parameters
    ----------
    mpos : list[MPO]
        The operators to combine
    weights : Optional[VectorLike]
        An optional sequence of weights to apply
    strategy : Strategy
        Truncation strategy when applying the MPO's.
    """

    __array_priority__ = 10000

    def __init__(
        self,
        mpos: Sequence[Union[MPO, MPOList]],
        weights: Optional[list[Weight]] = None,
        strategy: Strategy = DEFAULT_STRATEGY,
    ):
        self._mpos = list(mpos)
        assert len(self._mpos) >= 1
        first_mpo = self._mpos[0]
        if isinstance(first_mpo, MPO):
            self._size = (<MPO>first_mpo)._size
        elif isinstance(first_mpo, MPOList):
            self._size = (<MPOList>first_mpo)._size
        else:
            raise ValueError("Argument to MPOSum is neither MPO nor MPOList")
        self._weights = [1.0] * len(self._mpos) if weights is None else list(weights)
        self._strategy = strategy

    def copy(self) -> MPOSum:
        return MPOSum(self._mpos, self._weights, self._strategy)

    def __add__(self, A: Union[MPO, MPOList, MPOSum]):
        """Add an MPO or an MPOSum from the MPOSum."""
        if isinstance(A, MPO):
            new_weights = self._weights + [1]
            new_mpos = self._mpos + [A]
        elif isinstance(A, MPOList):
            new_weights = self._weights + [1]
            new_mpos = self._mpos + [A]
        elif isinstance(A, MPOSum):
            new_weights = self._weights + A._weights
            new_mpos = self._mpos + A._mpos
        else:
            raise TypeError(f"Cannot add an MPOSum to an object of type {type(A)}")
        return MPOSum(mpos=new_mpos, weights=new_weights, strategy=self._strategy)

    def __sub__(self, A: Union[MPO, MPOSum, MPOList]):
        """Subtract an MPO, MPOList or MPOSum from the MPOSum."""
        if isinstance(A, MPO):
            new_weights = self._weights + [-1]
            new_mpos = self._mpos + [A]
        elif isinstance(A, MPOList):
            new_weights = self._weights + [-1]
            new_mpos = self._mpos + [A]
        elif isinstance(A, MPOSum):
            new_weights = self._weights + list((-1) * np.asarray(A._weights))
            new_mpos = self._mpos + A._mpos
        else:
            raise TypeError(
                f"Cannot subtract an object of type {type(A)} from an MPOSum"
            )
        return MPOSum(mpos=new_mpos, weights=new_weights, strategy=self._strategy)

    def __mul__(self, n: Weight) -> MPOSum:
        """Multiply an MPOSum quantum state by an scalar n (MPOSum * n)"""
        # TODO: Find a simpler test that also keeps mypy happy
        # about the type of 'n' after this if. This problem is also
        # in MPO, MPS, MPSSum, etc.
        if not isinstance(n, (int, float, complex)):
            raise TypeError(f"Cannot multiply MPOSum by {n}")
        return MPOSum(
            mpos=self._mpos,
            weights=[n * weight for weight in self._weights],
            strategy=self._strategy,
        )

    def __rmul__(self, n: Union[MPO, MPOSum, MPOList]) -> MPOSum:
        """Multiply an MPOSum quantum state by an scalar n (MPOSum * n)"""
        if not isinstance(n, (int, float, complex)):
            raise Exception(f"Cannot multiply MPOSum by {n}")
        return MPOSum(
            mpos=self._mpos,
            weights=[n * weight for weight in self._weights],
            strategy=self._strategy,
        )

    def tomatrix(self) -> Operator:
        """Return the matrix representation of this MPO."""
        A = 0
        for i, mpo in enumerate(self._mpos):
            A = A + self._weights[i] * mpo.tomatrix()
        return A

    def set_strategy(self, strategy, strategy_components=None) -> MPOSum:
        """Return MPOSum with the given strategy."""
        if strategy_components is not None:
            mpos = [mpo.set_strategy(strategy_components) for mpo in self._mpos]
        else:
            mpos = self._mpos
        return MPOSum(mpos=mpos, weights=self._weights, strategy=strategy)

    def apply(
        self,
        state: Union[MPS, MPSSum],
        strategy: Optional[Strategy] = None,
    ) -> Union[MPS, MPSSum]:
        """Implement multiplication A @ state between an MPOSum 'A' and
        a Matrix Product State 'state'."""
        cdef Strategy the_strategy

        output: Union[MPS, MPSSum]
        for i, (w, O) in enumerate(zip(self._weights, self._mpos)):
            Ostate = w * O.apply(state)
            output = Ostate if i == 0 else output + Ostate

        the_strategy = self._strategy if strategy is None else strategy
        if the_strategy.simplify != SIMPLIFICATION_DO_NOT_SIMPLIFY:
            output = truncate.simplify(output, strategy=the_strategy)
        return output

    def __matmul__(self, b: Union[MPS, MPSSum]) -> Union[MPS, MPSSum]:
        """Implement multiplication A @ b between an MPOSum 'A' and
        a Matrix Product State 'b'."""
        return self.apply(b)

    def extend(self, L, sites=None, dimensions=2) -> MPOSum:
        """Enlarge an MPOSum so that it acts on a larger Hilbert space with 'L' sites.

        Parameters
        ----------
        L : int
            The new size for all MPOs.
        dimensions : int | list[int]
            If it is an integer, it is the dimension of the new sites.
            If it is a list, it is the dimension of all sites.
        sites  : list[int]
            Where to place the tensors of the original MPO.

        Returns
        ------
        MPOSum
            The extended operator.
        """
        return MPOSum(
            mpos=[
                mpo.extend(L, sites=sites, dimensions=dimensions) for mpo in self._mpos
            ],
            weights=self._weights,
            strategy=self._strategy,
        )

    def _joined_tensors(self, i: int, mpos: list[MPO]) -> Tensor4:
        """Join the tensors from all MPOs into bigger tensors."""
        As: list[Tensor4] = [mpo[i] for mpo in mpos]
        L = self._size
        if i == 0:
            return np.concatenate([w * A for w, A in zip(self._weights, As)], axis=-1)
        if i == L - 1:
            return np.concatenate(As, axis=0)

        DL: int = 0
        DR: int = 0
        d: int
        w: Weight = 0
        for A in As:
            a, d, d, b = A.shape
            DL += a
            DR += b
            w += A[0, 0, 0, 0]
        B = np.zeros((DL, d, d, DR), dtype=type(w))
        DL = 0
        DR = 0
        for A in As:
            a, d, d, b = A.shape
            B[DL : DL + a, :, :, DR : DR + b] = A
            DL += a
            DR += b
        return B

    def join(self, strategy: Optional[Strategy] = None) -> MPO:
        """Create an `MPO` by combining all tensors from all states in the linear
        combination.

        Returns
        -------
        MPS | CanonicalMPS
            Quantum state approximating this sum.
        """
        mpos = [m.join() if isinstance(m, MPOList) else m for m in self._mpos]
        return MPO(
            [self._joined_tensors(i, mpos) for i in range(self._size)],
            strategy=self._strategy if strategy is None else strategy,
        )

    def expectation(self, bra: MPS, ket: Optional[MPS] = None) -> Weight:
        """Expectation value of MPOList on one or two MPS states.

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
        return sum([m.expectation(bra, ket) for m in self._mpos])

    @property
    def mpos(self) -> list:
        return self._mpos

    @property
    def weights(self) -> list:
        return self._weights

    @property
    def strategy(self) -> Strategy:
        return self._strategy

    @property
    def size(self) -> int:
        return self._size
