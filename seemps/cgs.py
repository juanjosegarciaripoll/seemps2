from .typing import Optional
from .expectation import scprod
from .state import DEFAULT_TOLERANCE, MPS
from .mpo import MPO
from .truncate.combine import combine
from .tools import log


def cgs(
    A: MPO,
    b: MPS,
    guess: Optional[MPS] = None,
    maxiter: int = 100,
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
    tolerance : float, default = DEFAULT_TOLERANCE
        Truncation tolerance and error tolerance for the algorithm.

    Results
    -------
    MPS
        Approximate solution to :math:`A ψ = b`
    float
        Norm square of the residual :math:`\\Vert{A \\psi - b}\\Vert^2`
    """
    normb = scprod(b, b).real
    r = b
    if guess is not None:
        x: MPS = guess
        r = combine([1.0, -1.0], [b, A.apply(x)], tolerance=tolerance, normalize=False)
    p = r
    ρ = scprod(r, r).real
    log(f"CGS algorithm for {maxiter} iterations")
    for i in range(maxiter):
        Ap = A.apply(p)
        α = ρ / scprod(p, Ap).real
        if i > 0 or guess is not None:
            x = combine([1, α], [x, p], tolerance=tolerance, normalize=False)
        else:
            x = combine([α], [p], tolerance=tolerance, normalize=False)
        r = combine([1, -1], [b, A.apply(x)], tolerance=tolerance, normalize=False)
        ρ, ρold = scprod(r, r).real, ρ
        if ρ < tolerance * normb:
            log("Breaking on convergence")
            break
        p = combine([1.0, ρ / ρold], [r, p], tolerance=tolerance, normalize=False)
        log(f"Iteration {i:5}: |r|={ρ:5g}")
    return x, abs(ρ)
