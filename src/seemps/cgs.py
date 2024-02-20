from __future__ import annotations
from typing import Optional, Callable, Any, Union
from .state import (
    MPS,
    MPSSum,
    CanonicalMPS,
    DEFAULT_TOLERANCE,
    DEFAULT_STRATEGY,
    Strategy,
)
from .operators import MPO, MPOList, MPOSum
from .truncate import simplify
from . import tools


# TODO: Write tests for this
def cgs(
    A: Union[MPO, MPOList, MPOSum],
    b: Union[MPS, MPSSum],
    guess: Optional[MPS] = None,
    maxiter: int = 100,
    strategy: Strategy = DEFAULT_STRATEGY,
    tolerance: float = DEFAULT_TOLERANCE,
    callback: Optional[Callable[[MPS, float], Any]] = None,
) -> tuple[CanonicalMPS, float]:
    """Approximate solution of :math:`A \\psi = b`.

    Given the :class:`MPO` `A` and the :class:`MPS` `b`, use the conjugate
    gradient method to estimate another MPS that solves the linear system of
    equations :math:`A \\psi = b`.

    Parameters
    ----------
    A : MPO | MPOList | MPOSum
        Matrix product state that will be inverted
    b : MPS | MPSSum
        Right-hand side of the equation
    maxiter : int, default = 100
        Maximum number of iterations
    strategy : Strategy, default = DEFAULT_STRATEGY
        Truncation strategy for MPS and MPO operations
    tolerance : float, default = DEFAULT_TOLERANCE
        Error tolerance for the algorithm.

    Results
    -------
    MPS
        Approximate solution to :math:`A ψ = b`
    float
        Norm square of the residual :math:`\\Vert{A \\psi - b}\\Vert^2`
    """
    normb2 = b.norm_squared()
    if strategy.get_normalize_flag():
        strategy = strategy.replace(normalize=False)
    x = simplify(b if guess is None else guess, strategy=strategy)
    r = b - A @ x
    p = simplify(r, strategy=strategy)
    ρ = r.norm_squared()
    tools.log(f"CGS algorithm for {maxiter} iterations")
    for i in range(maxiter):
        α = ρ / A.expectation(p).real
        x = simplify(MPSSum([1, α], [x, p]), strategy=strategy)
        r = b - A @ x
        ρ, ρold = r.norm_squared(), ρ
        if callback is not None:
            callback(x, ρ)
        if ρ < tolerance * normb2:
            tools.log(
                f"CGS converged with residual {ρ} below relative tolerance {tolerance}"
            )
            break
        p = simplify(MPSSum([1.0, ρ / ρold], [r, p]), strategy=strategy)
        tools.log(f"Iteration {i:5}: |r|^2={ρ:5g} tol={tolerance:5g}")
    return x, abs(ρ)
