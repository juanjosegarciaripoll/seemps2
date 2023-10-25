from __future__ import annotations

import numpy as np
import seemps.truncate

from ..state import DEFAULT_STRATEGY, MPS, MPSSum, Strategy
from ..typing import *
from .mpo import MPO, MPOList


class MPOSum(object):
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

    mpos: list[Union[MPO, MPOList]]
    weights: list[Weight]
    size: int

    __array_priority__ = 10000

    def __init__(
        self,
        mpos: Sequence[Union[MPO, MPOList]],
        weights: Optional[list[Weight]] = None,
        strategy: Strategy = DEFAULT_STRATEGY,
    ):
        self.mpos = mpos = list(mpos)
        assert len(mpos) >= 1
        self.size = self.mpos[0].size
        self.weights = [1.0] * len(self.mpos) if weights is None else list(weights)
        self.strategy = strategy

    def __add__(self, A: Union[MPO, MPOList, MPOSum]):
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

    def __sub__(self, A: Union[MPO, MPOSum, MPOList]):
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

    def __mul__(self, n: Weight) -> MPOSum:
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

    def __rmul__(self, n: Union[MPO, MPOSum, MPOList]) -> MPOSum:
        """Multiply an MPOSum quantum state by an scalar n (MPOSum * n)"""
        if not isinstance(n, (int, float, complex)):
            raise Exception(f"Cannot multiply MPOSum by {n}")
        return MPOSum(
            mpos=self.mpos,
            weights=[n * weight for weight in self.weights],
            strategy=self.strategy,
        )

    def tomatrix(self) -> Operator:
        """Return the matrix representation of this MPO."""
        A = self.weights[0] * self.mpos[0].tomatrix()
        for i, mpo in enumerate(self.mpos[1:]):
            A = A + self.weights[i + 1] * mpo.tomatrix()
        return A

    def apply(
        self, 
        b: Union[MPS, MPSSum], 
        strategy: Optional[Strategy] = None, 
        simplify: Optional[bool] = None,
    ) -> Union[MPS, MPSSum]:
        """Implement multiplication A @ b between an MPOSum 'A' and
        a Matrix Product State 'b'."""
        # TODO: Is this really needed?
        if strategy is None:
            strategy = self.strategy
        if simplify is None:
            simplify = strategy.get_simplify_flag()
        if isinstance(b, MPSSum):
           state: MPS = seemps.truncate.simplify.combine(weights=b.weights, states=b.states, truncation=strategy)
        elif isinstance(b, MPS):
            state = b
        output: Union[MPS, MPSSum]
        for i, (w, O) in enumerate(zip(self.weights, self.mpos)):
            Ostate = w * O.apply(state, strategy=strategy)
            output = Ostate if i == 0 else output + Ostate
        if simplify:
            output = seemps.truncate.simplify(
                output, truncation=strategy
            )
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
        mpos = [m.join() if isinstance(m, MPOList) else m for m in self.mpos]
        return MPO(
            [self._joined_tensors(i, mpos) for i in range(self.size)],
            strategy=self.strategy if strategy is None else strategy,
        )
