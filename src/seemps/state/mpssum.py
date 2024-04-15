from __future__ import annotations
import numpy as np
from math import sqrt
from typing import Union, Iterable
from ..tools import InvalidOperation
from ..typing import Weight, Vector, Tensor3
from .core import MPSSum, CanonicalMPS, MPS
from .environments import scprod


def as_mps(self) -> MPS:
    return self.join()


def __add__(self, state: Union[MPS, MPSSum]) -> MPSSum:
    """Add `self + state`, incorporating it to the lists."""
    match state:
        case MPS():
            return MPSSum(self.weights + [1.0], self.states + [state], check_args=False)
        case MPSSum(weights=w, states=s):
            return MPSSum(self.weights + w, self.states + s, check_args=False)
        case _:
            raise InvalidOperation("+", self, state)


def __sub__(self, state: Union[MPS, MPSSum]) -> MPSSum:
    """Subtract `self - state`, incorporating it to the lists."""
    match state:
        case MPS():
            return MPSSum(self.weights + [-1], self.states + [state], check_args=False)
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


def join_canonical(self, *args, **kwdargs) -> CanonicalMPS:
    """Similar to join() but return canonical form"""
    return CanonicalMPS(self.join(), *args, **kwdargs)


MPSSum.__array_priority__ = 10000
MPSSum.as_mps = as_mps
MPSSum.join = join
MPSSum.join_canonical = join_canonical
MPSSum._joined_tensors = _joined_tensors
MPSSum.__add__ = __add__
MPSSum.__sub__ = __sub__
MPSSum.__mul__ = __mul__
MPSSum.__rmul__ = __rmul__
MPSSum.to_vector = to_vector

__all__ = ["MPSSum"]
