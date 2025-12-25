from __future__ import annotations
from typing import Callable, Any
from ..state import (
    MPS,
    MPSSum,
    CanonicalMPS,
    DEFAULT_TOLERANCE,
    DEFAULT_STRATEGY,
    Strategy,
    simplify,
)
from ..operators import MPO, MPOList, MPOSum
from ..tools import make_logger


def cgs_solve(
    A: MPO | MPOList | MPOSum,
    b: MPS | MPSSum,
    guess: MPS | None = None,
    maxiter: int = 100,
    tolerance: float = DEFAULT_TOLERANCE,
    strategy: Strategy = DEFAULT_STRATEGY,
    callback: Callable[[MPS, float], Any] | None = None,
) -> tuple[CanonicalMPS, float]:
    """Approximate solution of :math:`A \\psi = b`.

    Given the :class:`MPO` `A` and the :class:`MPS` `b`, use the conjugate
    gradient method to estimate another MPS that solves the linear system of
    equations :math:`A \\psi = b`. Convergence is determined by the
    residual :math:`\\Vert{A \\psi - b}\\Vert` being smaller than `tol`.

    Parameters
    ----------
    A : MPO | MPOList | MPOSum
        Matrix product state that will be inverted
    b : MPS | MPSSum
        Right-hand side of the equation
    maxiter : int, default = 100
        Maximum number of iterations
    tol : float, default = DEFAULT_TOLERANCE
        Error tolerance for the algorithm.
    strategy : Strategy, default = DEFAULT_STRATEGY
        Truncation strategy for MPS and MPO operations

    Returns
    -------
    MPS
        Approximate solution to :math:`A ψ = b`
    float
        Norm-2 of the residual :math:`\\Vert{A \\psi - b}\\Vert`
    """
    normb = b.norm()
    if strategy.get_normalize_flag():
        strategy = strategy.replace(normalize=False)
    x = simplify(b if guess is None else guess, strategy=strategy)
    r = b - A @ x
    p = simplify(r, strategy=strategy)
    residual = r.norm()
    with make_logger(2) as logger:
        logger(f"CGS algorithm for {maxiter} iterations", flush=True)
        for i in range(maxiter):
            if residual < tolerance * normb:
                logger(
                    f"CGS converged with residual {residual} below relative tolerance {tolerance}"
                )
                break
            α = residual * residual / A.expectation(p)
            x = simplify(MPSSum([1, α], [x, p]), strategy=strategy)
            r = b - A @ x
            residual, ρold = r.norm(), residual
            if callback is not None:
                callback(x, residual)
            p = simplify(MPSSum([1.0, residual / ρold], [r, p]), strategy=strategy)
            logger(f"CGS step {i:5}: |r|^2={residual:5g} tol={tolerance:5g}")
    return x, abs(residual)
