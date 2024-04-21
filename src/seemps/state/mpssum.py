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


MPSSum.as_mps = as_mps  # type: ignore
MPSSum.join = join  # type: ignore
MPSSum.join_canonical = join_canonical  # type: ignore
MPSSum._joined_tensors = _joined_tensors  # type: ignore
MPSSum.to_vector = to_vector  # type: ignore

__all__ = ["MPSSum"]
