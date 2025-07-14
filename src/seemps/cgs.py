from typing import Callable, Any
from .solve.cgs import cgs_solve
from .operators import MPO, MPOList, MPOSum
from .state import (
    MPS,
    CanonicalMPS,
    MPSSum,
    DEFAULT_TOLERANCE,
    DEFAULT_STRATEGY,
    Strategy,
)
import warnings


def cgs(
    A: MPO | MPOList | MPOSum,
    b: MPS | MPSSum,
    guess: MPS | None = None,
    maxiter: int = 100,
    tolerance: float = DEFAULT_TOLERANCE,
    strategy: Strategy = DEFAULT_STRATEGY,
    callback: Callable[[MPS, float], Any] | None = None,
) -> tuple[CanonicalMPS, float]:
    warnings.warn(
        "Use of deprecated seemps2.cgs. Please use seemps2.solve.cgs.cgs_solve instead"
    )
    return cgs_solve(A, b, guess, maxiter, tolerance, strategy, callback)


__all__ = ["cgs"]
