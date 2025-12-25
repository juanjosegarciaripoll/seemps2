from __future__ import annotations
import numpy as np
import warnings
from collections.abc import Sequence
from ..typing import Weight, DenseOperator, Tensor4
from ..state import DEFAULT_STRATEGY, MPS, MPSSum, Strategy
from .mpo import MPO, MPOList
from ..state import simplify_mps


class MPOSum(object):
    """Object representing a linear combination of matrix-product opeators.

    Parameters
    ----------
    mpos : list[MPO | MPOList]
        The operators to combine
    weights : VectorLike | None
        An optional sequence of weights to apply
    strategy : Strategy
        Truncation strategy when applying the MPO's.
    """

    mpos: list[MPO | MPOList]
    weights: list[Weight]
    size: int
    strategy: Strategy

    __array_priority__: int = 10000

    def __init__(
        self,
        mpos: Sequence[MPO | MPOList],
        weights: Sequence[Weight] | None = None,
        strategy: Strategy = DEFAULT_STRATEGY,
    ):
        self.mpos = mpos = list(mpos)
        assert len(mpos) >= 1
        self.size = self.mpos[0].size
        self.weights = [1.0] * len(mpos) if weights is None else list(weights)
        self.strategy = strategy

    # TODO: Rename to physical_dimensions()
    def dimensions(self) -> list[int]:
        """Return the physical dimensions of the MPO."""
        return self.mpos[0].dimensions()

    def copy(self) -> MPOSum:
        return MPOSum(self.mpos, self.weights, self.strategy)

    def __add__(self, A: MPO | MPOList | MPOSum):
        """Add an MPO or an MPOSum from the MPOSum."""
        if isinstance(A, MPO):
            new_weights = self.weights + [1]
            new_mpos = self.mpos + [A]
        elif isinstance(A, MPOList):
            new_weights = self.weights + [1]
            new_mpos = self.mpos + [A]
        elif isinstance(A, MPOSum):
            new_weights = self.weights + A.weights
            new_mpos = self.mpos + A.mpos
        else:
            raise TypeError(f"Cannot add an MPOSum to an object of type {type(A)}")
        return MPOSum(mpos=new_mpos, weights=new_weights, strategy=self.strategy)

    def __sub__(self, A: MPO | MPOList | MPOSum):
        """Subtract an MPO, MPOList or MPOSum from the MPOSum."""
        if isinstance(A, MPO):
            new_weights = self.weights + [-1]
            new_mpos = self.mpos + [A]
        elif isinstance(A, MPOList):
            new_weights = self.weights + [-1]
            new_mpos = self.mpos + [A]
        elif isinstance(A, MPOSum):
            new_weights = self.weights + list((-1) * np.asarray(A.weights))
            new_mpos = self.mpos + A.mpos
        else:
            raise TypeError(
                f"Cannot subtract an object of type {type(A)} from an MPOSum"
            )
        return MPOSum(mpos=new_mpos, weights=new_weights, strategy=self.strategy)

    def __rmul__(self, n: Weight) -> MPOSum:
        """Multiply an MPOSum quantum state by an scalar n (MPOSum * n)"""
        # TODO: Find a simpler test that also keeps mypy happy
        # about the type of 'n' after this if. This problem is also
        # in MPO, MPS, MPSSum, etc.
        if not isinstance(n, (int, float, complex)):
            raise TypeError(f"Cannot multiply MPOSum by {n}")
        return MPOSum(
            mpos=self.mpos,
            weights=[n * weight for weight in self.weights],
            strategy=self.strategy,
        )

    def __mul__(self, n: Weight) -> MPOSum:
        """Multiply an MPOSum operator by an scalar n (MPOSum * n)"""
        if not isinstance(n, (int, float, complex)):
            raise Exception(f"Cannot multiply MPOSum by {n}")
        return MPOSum(
            mpos=self.mpos,
            weights=[n * weight for weight in self.weights],
            strategy=self.strategy,
        )

    @property
    def T(self) -> MPOSum:
        """Return the transpose of this operator."""
        output = self.copy()
        output.mpos = [A.T for A in output.mpos]
        return output

    def tomatrix(self) -> DenseOperator:
        """Return the matrix representation of this MPO (Deprecated, see :meth:`to_matrix`)."""
        warnings.warn(
            "MPOSum.tomatrix() has been renamed to to_matrix()",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.to_matrix()

    def to_matrix(self) -> DenseOperator:
        """Return the matrix representation of this MPO."""
        A: DenseOperator = self.weights[0] * self.mpos[0].to_matrix()
        for i, mpo in enumerate(self.mpos[1:]):
            A = A + self.weights[i + 1] * mpo.to_matrix()
        return A

    def set_strategy(
        self, strategy: Strategy, strategy_components: Strategy | None = None
    ) -> MPOSum:
        """Return MPOSum with the given strategy."""
        if strategy_components is not None:
            mpos = [mpo.set_strategy(strategy_components) for mpo in self.mpos]
        else:
            mpos = self.mpos
        return MPOSum(mpos=mpos, weights=self.weights, strategy=strategy)

    def apply(
        self,
        state: MPS | MPSSum,
        strategy: Strategy | None = None,
        simplify: bool | None = None,
    ) -> MPS | MPSSum:
        """Implement multiplication A @ state between an MPOSum 'A' and
        a Matrix Product State 'state'."""
        output = MPSSum(
            [1] * len(self.weights),
            [w * O.apply(state) for w, O in zip(self.weights, self.mpos)],
        )
        # TODO: Is this really needed?
        if strategy is None:
            strategy = self.strategy
        if simplify is None:
            simplify = strategy.get_simplify_flag()
        if simplify:
            return simplify_mps(output, strategy=strategy)
        return output

    def __matmul__(self, b: MPS | MPSSum) -> MPS | MPSSum:
        """Implement multiplication A @ b between an MPOSum 'A' and
        a Matrix Product State 'b'."""
        return self.apply(b)

    def extend(
        self,
        L: int,
        sites: list[int] | None = None,
        dimensions: int | list[int] = 2,
    ) -> MPOSum:
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
        -------
        MPOSum
            The extended operator.
        """
        return MPOSum(
            mpos=[
                mpo.extend(L, sites=sites, dimensions=dimensions) for mpo in self.mpos
            ],
            weights=self.weights,
            strategy=self.strategy,
        )

    def _joined_tensors(self, i: int, mpos: list[MPO]) -> Tensor4:
        """Join the tensors from all MPOs into bigger tensors."""
        As: list[Tensor4] = [mpo[i] for mpo in mpos]
        L = self.size
        if i == 0:
            return np.concatenate([w * A for w, A in zip(self.weights, As)], axis=-1)
        if i == L - 1:
            return np.concatenate(As, axis=0)

        DL: int = 0
        DR: int = 0
        d: int = 0
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

    def join(self, strategy: Strategy | None = None) -> MPO:
        """Create an `MPO` by combining all tensors from all states in the linear
        combination.

        Returns
        -------
        MPS | CanonicalMPS
            Quantum state approximating this sum.
        """
        mpos = [m.join() if isinstance(m, MPOList) else m for m in self.mpos]
        return MPO(
            [self._joined_tensors(i, mpos) for i in range(self.size)],
            strategy=self.strategy if strategy is None else strategy,
        )

    def expectation(self, bra: MPS, ket: MPS | None = None) -> Weight:
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
        ket : MPS | None
            The ket component of the expectation value. Defaults to `bra`.

        Returns
        -------
        float | complex
            :math:`\\langle\\psi\\vert{O}\\vert\\phi\\rangle` where `O`
            is the matrix-product operator.
        """
        return sum([m.expectation(bra, ket) for m in self.mpos])

    def reverse(self) -> MPOSum:
        """Reverse the sites (see :meth:`~seemps.operators.MPO.reverse`)."""
        return MPOSum([o.reverse() for o in self.mpos], self.weights, self.strategy)
