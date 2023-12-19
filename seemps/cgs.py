from typing import Optional, Union
from .expectation import scprod
from .state import MPS, MPSSum, DEFAULT_TOLERANCE, DEFAULT_STRATEGY, Strategy
from .mpo import MPO
from .truncate.combine import combine
from .tools import log


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
    r = b
    if strategy.get_normalize_flag():
        strategy = strategy.replace(normalize=False)
    if guess is not None:
        Ax: MPS = A.apply(guess)  # type: ignore
        r = combine([1.0, -1.0], [b, Ax], strategy=strategy)
    p = r
    ρ = scprod(r, r).real
    log(f"CGS algorithm for {maxiter} iterations")
    x: MPS
    for i in range(maxiter):
        Ap: MPS = A.apply(p)  # type: ignore
        α = ρ / scprod(p, Ap).real
        if i > 0 or guess is not None:
            x = combine([1, α], [x, p], strategy=strategy)
        else:
            x = combine([α], [p], strategy=strategy)
        Ax: MPS = A.apply(guess)  # type: ignore
        r = combine([1, -1], [b, Ax], strategy=strategy)
        ρ, ρold = scprod(r, r).real, ρ
        if ρ < tolerance * normb:
            log("Breaking on convergence")
            break
        p = combine([1.0, ρ / ρold], [r, p], strategy=strategy)
        log(f"Iteration {i:5}: |r|={ρ:5g}")
    return x, abs(ρ)
