from typing import Optional
from .expectation import scprod
from .state import MPS, MPSSum, DEFAULT_TOLERANCE, DEFAULT_STRATEGY, Strategy
from .mpo import MPO
from .truncate import simplify
from . import tools


# TODO: Write tests for this
def cgs(
    A: MPO,
    b: MPS,
    guess: Optional[MPS] = None,
    maxiter: int = 100,
    strategy: Strategy = DEFAULT_STRATEGY,
    tolerance: float = DEFAULT_TOLERANCE,
) -> tuple[MPS, float]:
    """Approximate solution of :math:`A \\psi = b`.

    Given the :class:`MPO` `A` and the :class:`MPS` `b`, use the conjugate
    gradient method to estimate another MPS that solves the linear system of
    equations :math:`A \\psi = b`.

    Parameters
    ----------
    A : MPO
        Matrix product state that will be inverted
    b : MPS
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
    normb = scprod(b, b).real
    if strategy.get_normalize_flag():
        strategy = strategy.replace(normalize=False)
    x = b if guess is None else guess
    r = simplify(MPSSum([1.0, -1.0], [b, A @ x]), strategy=strategy)
    p = r
    ρ = scprod(r, r).real
    tools.log(f"CGS algorithm for {maxiter} iterations")
    for i in range(maxiter):
        α = ρ / A.expectation(p).real
        x = simplify(MPSSum([1, α], [x, p]), strategy=strategy)
        r = simplify(MPSSum([1, -1.0], [b, A @ x]), strategy=strategy)
        ρ, ρold = scprod(r, r).real, ρ
        if ρ < tolerance * normb:
            tools.log(
                f"CGS converged with residual {ρ} below relative tolerance {tolerance}"
            )
            break
        p = simplify(MPSSum([1.0, ρ / ρold], [r, p]), strategy=strategy)
        tools.log(f"Iteration {i:5}: |r|^2={ρ:5g} tol={tolerance:5g}")
    return x, abs(ρ)
