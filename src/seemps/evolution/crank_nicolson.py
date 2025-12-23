from __future__ import annotations
import numpy as np
from ..analysis.operators import id_mpo
from ..solve import cgs_solve
from ..operators import MPO, MPOSum
from ..state import DEFAULT_STRATEGY, MPS, Strategy
from .common import ODECallback, TimeSpan, ode_solver


def crank_nicolson(
    H: MPO,
    time: TimeSpan,
    state: MPS,
    steps: int = 1000,
    tol_cgs: float = 1e-7,
    maxiter_cgs: int = 50,
    strategy: Strategy = DEFAULT_STRATEGY,
    callback: ODECallback | None = None,
    itime: bool = False,
):
    r"""Solve a Schrodinger equation using a fourth order Runge-Kutta method.

    See :func:`seemps.evolution.euler` for a description of the
    missing function arguments and the function's output.

    Parameters
    ----------
    tol_cgs: float
        Tolerance of the CGS algorithm.
    maxiter_cgs: int
        Maximum number of iterations of the CGS algorithm.
    """
    A: MPO | None = None
    B: MPO | None = None
    last_dt: float = np.inf
    id = id_mpo(state.size, strategy=H.strategy)

    def evolve_for_dt(
        t: float,
        state: MPS,
        factor: complex | float,
        dt: float,
        normalize_strategy: Strategy,
    ) -> MPS:
        nonlocal A, B, last_dt
        if last_dt != dt or A is None or B is None:
            last_dt = dt
            idt = factor * dt
            A = MPOSum(mpos=[id, H], weights=[1, 0.5 * idt], strategy=H.strategy).join(
                strategy=H.strategy
            )
            B = MPOSum(mpos=[id, H], weights=[1, -0.5 * idt], strategy=H.strategy).join(
                strategy=H.strategy
            )
        # TODO: Consider using dmrg_solve or other algorithm
        state, _ = cgs_solve(
            A,
            B @ state,
            guess=state,
            tolerance=tol_cgs,
            strategy=normalize_strategy,
            maxiter=maxiter_cgs,
        )
        return state

    return ode_solver(evolve_for_dt, time, state, steps, strategy, callback, itime)
