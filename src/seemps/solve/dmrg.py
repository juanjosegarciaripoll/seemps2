from __future__ import annotations
from math import sqrt
from typing import Callable
import numpy as np
import scipy.sparse.linalg
from ..tools import make_logger
from ..typing import Tensor4
from ..state import DEFAULT_STRATEGY, MPS, CanonicalMPS, Strategy, scprod
from ..state.simplification import AntilinearForm
from ..cython import _contract_last_and_first
from ..operators import MPO
from ..operators.quadratic import QuadraticForm


SolverFn = Callable[..., tuple[np.ndarray, int]]

_SOLVERS: dict[str, SolverFn] = {
    "cg": scipy.sparse.linalg.cg,
    "bicg": scipy.sparse.linalg.bicg,
    "gmres": scipy.sparse.linalg.gmres,
    "lgmres": scipy.sparse.linalg.lgmres,
    "bicgstab": scipy.sparse.linalg.bicgstab,
}


def _relative_change(a: MPS, b: MPS) -> float:
    a_norm_sq = a.norm_squared()
    b_norm_sq = b.norm_squared()
    if a_norm_sq == 0:
        if b_norm_sq == 0:
            return 0.0
        else:
            return np.inf

    d_sq = a_norm_sq - 2.0 * scprod(a, b).real + b_norm_sq
    return sqrt(max(d_sq, 0.0) / a_norm_sq)


def _solve_local(
    QF: QuadraticForm,
    i: int,
    b_tensor: Tensor4,
    atol: float,
    rtol: float,
    solver: SolverFn,
) -> Tensor4:
    """Solve the local two-site linear system at bond `i`."""
    Op = QF.two_site_operator(i)
    v = _contract_last_and_first(QF.state[i], QF.state[i + 1])
    b_flat = b_tensor.reshape(-1)
    x0 = v.reshape(-1)
    # Early escape
    residual_norm = float(np.linalg.norm(Op @ x0 - b_flat))
    tol = max(atol, rtol * float(np.linalg.norm(b_flat)))
    if residual_norm <= tol:
        return v
    x, _ = solver(Op, b_flat, x0, atol=atol, rtol=rtol)
    return x.reshape(v.shape)


def _sweep(
    QF: QuadraticForm,
    LF: AntilinearForm,
    direction: int,
    atol: float,
    rtol: float,
    solver: SolverFn,
    strategy: Strategy,
) -> None:
    """One full two-site sweep updating `QF` and `LF` in place."""
    size = QF.state.size
    sites = range(size - 1) if direction > 0 else range(size - 2, -1, -1)
    for i in sites:
        AB = _solve_local(QF, i, LF.tensor2site(direction), atol, rtol, solver)
        if direction > 0:
            QF.update_2site_right(AB, i, strategy)
            LF.update_right()
        else:
            QF.update_2site_left(AB, i, strategy)
            LF.update_left()


def dmrg_solve(
    A: MPO,
    b: MPS,
    guess: MPS | None = None,
    maxiter: int = 20,
    atol: float = 0,
    rtol: float = 1e-5,
    strategy: Strategy = DEFAULT_STRATEGY,
    method: str = "bicgstab",
    compute_residuals: bool = True,
) -> tuple[MPS, float | None]:
    r"""Solve :math:`A x = b` for an MPO `A` and an MPS `b` using two-site DMRG.

    Parameters
    ----------
    A : MPO
        Linear operator on the left-hand side.
    b : MPS
        Right-hand side.
    guess : MPS | None, default = None
        Initial guess (defaults to `b`).
    maxiter : int, default = 20
        Maximum number of sweeps.
    atol, rtol : float
        Convergence tolerance. With ``compute_residuals=True``, convergence
        requires ``norm(A@x - b) <= max(rtol * norm(b), atol)``. With
        ``compute_residuals=False``, the relative change of the solution
        between sweeps is used instead. Defaults are `rtol=1e-5` and `atol=0`.
    strategy : Strategy, default = DEFAULT_STRATEGY
        Truncation strategy for the bond dimension.
    method : str, deafult = 'bicgstab'
        Iterative solver: ``'cg'``, ``'bicg'``, ``'bicgstab'``, ``'gmres'``,
        or ``'lgmres'``.
    compute_residuals : bool, default = True
        If True, check the full residual at each sweep for convergence (robust).
        If False, use a cheaper relative-change criterion (faster).

    Returns
    -------
    MPS
        The solution `x`.
    float | None
        Residual :math:``\Vert A x - b\Vert_2``, or None when ``compute_residuals=False``.
    """
    if maxiter < 1:
        raise ValueError("maxiter must be positive")
    if guess is None:
        guess = b.copy()
    solver = _SOLVERS.get(method)
    if solver is None:
        raise ValueError(f'Unknown solver "{method}"')

    b_norm = b.norm()
    tol = max(atol, rtol * b_norm)
    strat = strategy.replace(normalize=False)
    logger = make_logger()
    logger(f"DMRG solver initiated with maxiter={maxiter}, tolerance={tol}")

    if not isinstance(guess, CanonicalMPS) or guess.center not in (0, A.size - 1):
        guess = CanonicalMPS(guess, center=0, strategy=strat)
    if guess.center == 0:
        direction = +1
        QF = QuadraticForm(A, guess, start=0)
        LF = AntilinearForm(guess, b, center=0)
    else:
        direction = -1
        QF = QuadraticForm(A, guess, start=A.size - 2)
        LF = AntilinearForm(guess, b, center=A.size - 1)

    residual: float | None = np.inf
    change_tol = max(rtol, atol / b_norm) if b_norm > 0 else rtol
    if compute_residuals:
        residual = (A @ QF.state - b).norm()
        logger(f"initial residual={residual}")
        if residual <= tol:
            logger(f"Converged below tolerance {tol}")
            logger.close()
            return QF.state, residual

    psi_old: MPS | None = None
    for sweep in range(1, maxiter + 1):
        if not compute_residuals:
            psi_old = QF.state.copy()
        _sweep(QF, LF, direction, atol, rtol, solver, strat)
        direction = -direction

        if compute_residuals:
            residual = (A @ QF.state - b).norm()
            logger(f"sweep={sweep}, residual={residual}")
            if residual <= tol:
                logger(f"Converged below tolerance {tol}")
                break
        else:
            if psi_old is None:
                raise RuntimeError("psi_old was not initialized before comparison")
            change = _relative_change(QF.state, psi_old)
            logger(f"sweep={sweep}, relative_change={change}")
            if change <= change_tol:
                logger(f"Converged below tolerance {change_tol}")
                break

    logger.close()
    if not compute_residuals:
        residual = None

    return QF.state, residual
