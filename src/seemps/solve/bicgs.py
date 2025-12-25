from __future__ import annotations
from ..state import (
    MPS,
    MPSSum,
    CanonicalMPS,
    scprod,
    DEFAULT_STRATEGY,
    Strategy,
    simplify,
)
from ..operators import MPO, MPOList, MPOSum
from ..tools import make_logger


# TODO: Write tests for this
def bicgs_solve(
    A: MPO | MPOList | MPOSum,
    b: MPS | MPSSum,
    guess: MPS | None = None,
    maxiter: int = 100,
    atol: float = 0.0,
    rtol: float = 1e-5,
    strategy: Strategy = DEFAULT_STRATEGY,
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
    atol, rtol : float
        Absolute and relative tolerance for the convergence of the algorithm.
        `norm(A@x - b) <= max(rtol * norm(b), atol)`. Defaults are
        `rtol=1e-5` and `atol=0`
    strategy : Strategy, default = DEFAULT_STRATEGY
        Truncation strategy to keep bond dimensions in check. Defaults to
        `DEFAULT_STRATEGY`, which is very strict.

    Returns
    -------
    MPS
        Approximate solution to :math:`A ψ = b`
    float
        Norm square of the residual :math:`\\Vert{A \\psi - b}\\Vert^2`
    """
    normb = b.norm()
    tolerance = max(rtol * normb, atol)
    x = simplify(b if guess is None else guess, strategy=strategy)
    p = r = r0 = simplify(b - A @ x, strategy)
    norm_r = rho = r0.norm()
    with make_logger(2) as logger:
        logger(f"BICCGS algorithm for {maxiter} iterations", flush=True)
        if norm_r < tolerance:
            logger(
                f"BICCGS converged with residual {norm_r} below tolerance {tolerance}"
            )
            return x, norm_r
        for _ in range(1, maxiter + 1):
            v = simplify(A @ p, strategy)
            alpha = rho / scprod(r0, v)
            h = simplify(x + alpha * p, strategy)
            s = simplify(r - alpha * v, strategy)
            residual = s.norm()
            if residual < tolerance:
                logger(
                    f"BICCGS converged with residual {residual} below tolerance {tolerance}"
                )
                x = h
                break
            t = simplify(A @ s, strategy)
            w = scprod(t, s) / t.norm_squared()
            x = simplify(h + w * s, strategy)
            r = simplify(s - w * t, strategy)
            norm_r = r.norm()
            if norm_r < tolerance:
                logger(
                    f"BICCGS converged with residual {norm_r} below tolerance {tolerance}"
                )
                break
            rho_new = scprod(r0, r)
            beta = (rho_new / rho) * (alpha / w)
            rho = abs(rho_new)
            p = simplify(r + beta * p - (beta * w) * v, strategy)

    return x, norm_r  # Not converged within max_iter
