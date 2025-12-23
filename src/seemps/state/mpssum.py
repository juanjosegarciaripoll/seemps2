from __future__ import annotations
import numpy as np
from math import sqrt
from collections.abc import Iterable, Sequence
from typing import cast
from ..tools import InvalidOperation
from ..typing import Weight, Vector, Tensor3
from .environments import scprod


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
    __array_priority__: int = 10000

    def __init__(
        self,
        weights: Iterable[Weight],
        states: Iterable[MPS | MPSSum],
        check_args: bool = True,
    ):
        if check_args:
            self.weights = new_weights = []
            self.states = new_states = []
            for w, s in zip(weights, states):
                if isinstance(s, MPS):
                    new_weights.append(w)
                    new_states.append(s)
                elif isinstance(s, MPSSum):
                    new_weights.extend(w * wi for wi in s.weights)
                    new_states.extend(s.states)
                else:
                    raise ValueError(s)
            self.size = new_states[0].size
        else:
            self.weights = list(weights)
            self.states = cast(list[MPS], list(states))
            self.size = self.states[0].size

    def as_mps(self) -> MPS:
        return self.join()

    def copy(self) -> MPSSum:
        """Return a shallow copy of the MPS sum and its data. Does not copy
        the states, only the list that stores them."""
        return MPSSum(self.weights.copy(), self.states.copy())

    def __copy__(self) -> MPSSum:
        return self.copy()

    def __add__(self, state: MPS | MPSSum) -> MPSSum:
        """Add `self + state`, incorporating it to the lists."""
        match state:
            case MPS():
                return MPSSum(
                    self.weights + [1.0], self.states + [state], check_args=False
                )
            case MPSSum(weights=w, states=s):
                return MPSSum(self.weights + w, self.states + s, check_args=False)
            case _:
                raise InvalidOperation("+", self, state)

    def __sub__(self, state: MPS | MPSSum) -> MPSSum:
        """Subtract `self - state`, incorporating it to the lists."""
        match state:
            case MPS():
                return MPSSum(
                    self.weights + [-1], self.states + [state], check_args=False
                )
            case MPSSum(weights=w, states=s):
                return MPSSum(
                    self.weights + [-wi for wi in w], self.states + s, check_args=False
                )
            case _:
                raise InvalidOperation("-", self, state)

    def __mul__(self, n: Weight) -> MPSSum:
        """Rescale the linear combination `n * self` for scalar `n`."""
        if isinstance(n, (int, float, complex)):
            return MPSSum([n * w for w in self.weights], self.states, check_args=False)
        raise InvalidOperation("*", self, n)

    def __rmul__(self, n: Weight) -> MPSSum:
        """Rescale the linear combination `self * n` for scalar `n`."""
        if isinstance(n, (int, float, complex)):
            return MPSSum([n * w for w in self.weights], self.states, check_args=False)
        raise InvalidOperation("*", n, self)

    def to_vector(self) -> Vector:
        """Return the wavefunction of this quantum state."""
        return sum(wa * A.to_vector() for wa, A in zip(self.weights, self.states))  # type: ignore # pyright: ignore[reportReturnType]

    def _joined_tensors(self, i: int, L: int) -> Tensor3:
        """Join the tensors from all MPS into bigger tensors."""
        As: list[Tensor3] = [s[i] for s in self.states]
        if i == 0:
            return np.concatenate([w * A for w, A in zip(self.weights, As)], axis=2)
        if i == L - 1:
            return np.concatenate(As, axis=0)

        DL: int = 0
        DR: int = 0
        d: int = 0
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

    def join(self) -> MPS:
        """Create an `MPS` by combining all tensors from all states in
        the linear combination.

        Returns
        -------
        MPS
            Quantum state approximating this sum.
        """
        L = self.size
        return MPS([self._joined_tensors(i, L) for i in range(L)])

    def join_canonical(self, *args, **kwdargs) -> CanonicalMPS:  # pyright: ignore[reportMissingParameterType]
        """Similar to join() but return canonical form"""
        return CanonicalMPS(self.join(), *args, **kwdargs)

    def conj(self) -> MPSSum:
        """Return the complex-conjugate of this quantum state."""
        return MPSSum(
            [np.conj(w) for w in self.weights], [state.conj() for state in self.states]
        )

    def norm_squared(self) -> float:
        """Norm-2 squared :math:`\\Vert{\\psi}\\Vert^2` of this MPS."""
        w = self.weights
        s = self.states
        L = len(w)
        return abs(
            sum(
                (w[i].conjugate() * w[j] * scprod(s[i], s[j])).real
                * (1 if i == j else 2)
                for i in range(L)
                for j in range(i, L)
            )
        )

    def norm(self) -> float:
        """Norm-2 :math:`\\Vert{\\psi}\\Vert^2` of this MPS."""
        return sqrt(self.norm_squared())

    def error(self) -> float:
        """Upper bound of the norm-2 error accumulated in this MPS."""
        return sum(
            abs(weight) * state._error
            for weight, state in zip(self.weights, self.states)
        )

    def delete_zero_components(self) -> float:
        """Compute the norm-squared of the linear combination of weights and
        states and eliminate states that are zero or have zero weight."""
        c: float = 0.0
        final_weights: list[Weight] = []
        final_states: list[MPS] = []
        for wi, si in zip(self.weights, self.states):
            wic = wi.conjugate()
            ni = (wic * wi).real * si.norm_squared()
            if ni:
                for wj, sj in zip(final_weights, final_states):
                    c += 2 * (wic * wj * scprod(si, sj)).real
                final_states.append(si)
                final_weights.append(wi)
                c += ni
        L = len(final_weights)
        if L < len(self.states):
            if L == 0:
                self.weights = [0.0]
                self.states = [self.states[0].zero_state()]
                return 0.0
            else:
                self.weights = final_weights
                self.states = final_states
        return abs(c)

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
    ) -> MPSSum:
        return MPSSum(
            self.weights,
            [A.extend(L, sites, dimensions, state) for A in self.states],
            check_args=False,
        )

    def reverse(self) -> MPSSum:
        """Reverse the sites (see :meth:`~seemps.state.MPS.reverse`)."""
        return MPSSum(
            self.weights[-1::],
            [state.reverse() for state in reversed(self.states)],
        )


from .canonical_mps import CanonicalMPS  # noqa: E402
from .mps import MPS  # noqa: E402


def to_mps(mps_or_sum: MPS | MPSSum) -> MPS:
    if isinstance(mps_or_sum, MPSSum):
        return mps_or_sum.join()
    return mps_or_sum
