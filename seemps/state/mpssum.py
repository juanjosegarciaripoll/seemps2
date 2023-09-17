from __future__ import annotations
import numpy as np
from ..typing import *
from ..tools import InvalidOperation
from .environments import *
from .schmidt import vector2mps
from .core import DEFAULT_STRATEGY, Strategy


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
        assert len(states) == len(weights)
        assert len(states) > 0
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
        strategy : Strategy
            Parameters for the simplificaiton and truncation algorithms.
            Defaults to `self.strategy`.

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
        strategy: Optional[Strategy] = None,
    ):
        """Create an `MPS` or `CanonicalMPS` state by combining all tensors
        from all states in the linear combination.

        Parameters
        ----------
        canonical: bool
            Whether to create the state in canonical form. Defaults to `True`.
        center: Optional[int]
            Center for the `CanonicalMPS`, if `canonical` is true.
        strategy: Strategy
            Parameters for the truncation algorithms used when creating the
            `CanonicalMPS`. Only used if `canonical` is `True`.
            Defaults to `self.strategy`.

        Returns
        -------
        MPS | CanonicalMPS
            Quantum state approximating this sum.
        """
        L = self.states[0].size
        data = [self._joined_tensors(i, L) for i in range(L)]
        if canonical:
            return CanonicalMPS(
                data,
                strategy=self.strategy if strategy is None else strategy,
                center=center,
            )
        else:
            return MPS(data)

    def conj(self) -> MPSSum:
        """Return the complex-conjugate of this quantum state."""
        return MPSSum(
            [np.conj(w) for w in self.weights],
            [state.conj() for state in self.states],
            self.strategy,
        )


from .mps import MPS
from .canonical_mps import CanonicalMPS
