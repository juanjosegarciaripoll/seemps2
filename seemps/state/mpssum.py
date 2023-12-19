from __future__ import annotations

import numpy as np

from ..tools import InvalidOperation
from ..typing import *
from .core import DEFAULT_STRATEGY, Strategy
from .environments import *
from .schmidt import vector2mps


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
    """

    weights: list[Weight]
    states: list[MPS]
    size: int

    #
    # This class contains all the matrices and vectors that form
    # a Matrix-Product State.
    #
    __array_priority__ = 10000

    def __init__(
        self,
        weights: list[Weight],
        states: list[MPS],
    ):
        # TODO: This is not consistent with MPS, MPO and MPOSum
        # which copy their input lists. We should decide whether we
        # want to copy or not.
        assert len(states) == len(weights)
        assert len(states) > 0
        self.weights = weights
        self.states = states
        self.size = states[0].size

    def copy(self) -> MPSSum:
        """Return a shallow copy of the MPS sum and its data. Does not copy
        the states, only the list that stores them."""
        return MPSSum(self.weights.copy(), self.states.copy())

    def __copy__(self) -> MPSSum:
        return self.copy()

    def __add__(self, state: Union[MPS, MPSSum]) -> MPSSum:
        """Add `self + state`, incorporating it to the lists."""
        if isinstance(state, MPS):
            return MPSSum(
                self.weights + [1.0],
                self.states + [state],
            )
        elif isinstance(state, MPSSum):
            return MPSSum(
                self.weights + state.weights,
                self.states + state.states,
            )
        raise InvalidOperation("+", self, state)

    def __sub__(self, state: Union[MPS, MPSSum]) -> MPSSum:
        """Subtract `self - state`, incorporating it to the lists."""
        if isinstance(state, MPS):
            return MPSSum(self.weights + [-1], self.states + [state])
        if isinstance(state, MPSSum):
            return MPSSum(
                self.weights + [-w for w in state.weights],
                self.states + state.states,
            )
        raise InvalidOperation("-", self, state)

    def __mul__(self, n: Weight) -> MPSSum:
        """Rescale the linear combination `n * self` for scalar `n`."""
        if isinstance(n, (int, float, complex)):
            return MPSSum([n * w for w in self.weights], self.states)
        raise InvalidOperation("*", self, n)

    def __rmul__(self, n: Weight) -> MPSSum:
        """Rescale the linear combination `self * n` for scalar `n`."""
        if isinstance(n, (int, float, complex)):
            return MPSSum([n * w for w in self.weights], self.states)
        raise InvalidOperation("*", n, self)

    def to_vector(self) -> Vector:
        """Return the wavefunction of this quantum state."""
        return sum(wa * A.to_vector() for wa, A in zip(self.weights, self.states))  # type: ignore

    def _joined_tensors(self, i: int, L: int) -> Tensor3:
        """Join the tensors from all MPS into bigger tensors."""
        As: list[Tensor3] = [s[i] for s in self.states]
        if i == 0:
            return np.concatenate([w * A for w, A in zip(self.weights, As)], axis=2)
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
        L = self.size
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
        return MPSSum(
            [np.conj(w) for w in self.weights], [state.conj() for state in self.states]
        )


from .canonical_mps import CanonicalMPS
from .mps import MPS
