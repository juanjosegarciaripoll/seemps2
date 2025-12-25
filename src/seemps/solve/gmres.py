from __future__ import annotations
import numpy as np
from ..typing import Float
from ..state import (
    MPS,
    MPSSum,
    CanonicalMPS,
    scprod,
    DEFAULT_STRATEGY,
    DEFAULT_TOLERANCE,
    Strategy,
    simplify,
)
from ..operators import MPO
from ..tools import make_logger


def gmres_solve(
    A: MPO,
    b: MPS,
    guess: MPS | None = None,
    nvectors: int = 5,
    max_restarts: int = 5,
    tolerance: float = DEFAULT_TOLERANCE,
    tol_ill_conditioning: Float = np.finfo(float).eps * 10,  # type: ignore
    strategy: Strategy = DEFAULT_STRATEGY,
) -> tuple[CanonicalMPS, float]:
    """Approximate solution of :math:`A \\psi = b`.

    Given the :class:`MPO` `A` and the :class:`MPS` `b`, use the generalized minimal resudial (GMRES) method to estimate another MPS that solves the linear system of equations :math:`A \\psi = b`.
    Convergence is determined by the residual :math:`\\Vert{A \\psi - b}\\Vert` being smaller than `tolerance`.

    Parameters
    ----------
    A : MPO
        The linear operator on the left-hand side of the equation.
    b : MPS
        The right-hand side vector.
    guess : MPS | None, default = None
        Initial guess for the solution. If None, uses zero.
    nvectors : int, default = 5
        Maximum size of the Krylov subspace at each restart.
    max_restarts : int, default = 5
        Maximum number of restarts.
    tolerance : float, default = DEFAULT_TOLERANCE
        Convergence tolerance for the residual norm.
    tol_ill_conditioning : float, default = np.finfo(float).eps * 10
        Tolerance for detecting ill-conditioning in the Krylov basis.
    strategy : Strategy, default = DEFAULT_STRATEGY
        Truncation strategy for MPS operations.

    Returns
    -------
    CanonicalMPS
        Approximate solution to :math:`A \\psi = b`.
    float
        Norm-2 of the residual :math:`\\Vert{A \\psi - b}\\Vert`.


    """
    normb = b.norm()
    if strategy.get_normalize_flag():
        strategy = strategy.replace(normalize=False)
    if guess is None:
        guess = b.zero_state()
    x = simplify(guess, strategy=strategy)

    # Residual
    r = simplify(b - A @ x, strategy=strategy)
    residual = r.norm()
    dtype = type(A[0][0, 0, 0, 0] * x[0][0, 0, 0] * b[0][0, 0, 0])

    with make_logger(2) as logger:
        logger(f"GMRES algorithm with {max_restarts=}, {nvectors=}", flush=True)

        for restart in range(max_restarts):
            # Check convergence
            if residual < tolerance * normb:
                logger(f"GMRES converged at restart {restart} with residual {residual}")
                break

            # Build Krylov subspace
            H = np.zeros((nvectors + 1, nvectors), dtype=dtype)
            V = [r * (1 / residual)]
            for j in range(nvectors):
                w = simplify(A @ V[j], strategy=strategy)

                # Modified Gram-Schmidt
                for i in range(j + 1):
                    hij = scprod(V[i], w)
                    H[i, j] = hij
                    w = simplify(w - hij * V[i], strategy=strategy)

                # Fill H matrix
                hj1 = w.norm()
                H[j + 1, j] = hj1

                # Check for ill-conditioning
                if hj1 < tol_ill_conditioning:
                    logger(f"Ill-conditioning detected at vector {j} with norm {hj1}")
                    break

                # Add vector to Krylov basis
                V.append(w * (1 / hj1))

            # Solve least squares problem in Krylov subspace
            m = max(len(V) - 1, 1)
            Hm = H[: m + 1, :m]
            rhs = np.zeros(m + 1)
            rhs[0] = residual
            y, *_ = np.linalg.lstsq(Hm, rhs, rcond=None)

            # Update solution
            x = simplify(x + MPSSum(y, V[:m]), strategy=strategy)
            r = simplify(b - A @ x, strategy=strategy)
            residual = r.norm()

            logger(f"GMRES restart {restart}: residual={residual}")

    return x, residual
