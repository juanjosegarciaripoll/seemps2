from __future__ import annotations
from typing import Callable
import numpy as np
from ..analysis.operators import id_mpo
from ..cgs import cgs
from ..operators import MPO, MPOSum
from ..state import DEFAULT_STRATEGY, MPS, Strategy
from ..typing import Vector


def crank_nicolson(
    H: MPO,
    t_span: float | tuple[float, float] | Vector,
    state: MPS,
    steps: int = 1000,
    tol_cgs: float = 1e-14,
    maxiter_cgs: int = 50,
    strategy: Strategy = DEFAULT_STRATEGY,
    callback: Callable | None = None,
    itime: bool = False,
):
    r"""Solve a Schrodinger equation using a fourth order Runge-Kutta method.

    See :function:`seemps.evolution.euler` for a description of the
    function arguments.

    Parameters
    ----------
    H : MPO
        Hamiltonian in MPO form.
    t_span : float | tuple[float, float] | Vector
        Integration interval, or sequence of time steps.
    state : MPS
        Initial guess of the ground state.
    steps : int, default = 1000
        Integration steps, if not defined by `t_span`.
    tol_cgs: float
        Tolerance of the CGS algorithm.
    maxiter_cgs: int
        Maximum number of iterations of the CGS algorithm.
    strategy : Strategy, default = DEFAULT_STRATEGY
        Truncation strategy for MPO and MPS algebra.
    callback : Callable[[float, MPS], Any] | None
        A callable called after each iteration (defaults to None).
    itime : bool, default = False
        Whether to solve the imaginary time evolution problem.

    Results
    -------
    result : MPS | list[Any]
        Final state after evolution or values collected by callback
    """
    if isinstance(t_span, (int, float)):
        t_span = (0.0, t_span)
    if len(t_span) == 2:
        t_span = np.linspace(t_span[0], t_span[1], steps + 1)
    factor: float | complex
    if itime:
        factor = 1
        normalize_strategy = strategy.replace(normalize=True)
    else:
        factor = 1j
        normalize_strategy = strategy
    last_t = t_span[0]
    output = []
    idt = factor * (t_span[1] - last_t)
    id = id_mpo(state.size, strategy=H.strategy)
    A = MPOSum(mpos=[id, H], weights=[1, 0.5 * idt], strategy=H.strategy).join(
        strategy=H.strategy
    )
    B = MPOSum(mpos=[id, H], weights=[1, -0.5 * idt], strategy=H.strategy).join(
        strategy=H.strategy
    )
    for t in t_span:
        if t != last_t:
            state, _ = cgs(
                A,
                B @ state,
                guess=state,
                tolerance=tol_cgs,
                strategy=normalize_strategy,
                maxiter=maxiter_cgs,
            )
        if callback is not None:
            output.append(callback(t, state))
        last_t = t
    if callback is None:
        return state
    else:
        return output
