from __future__ import annotations
import numpy as np
from typing import Union, Optional, Callable
from ..state import MPS, CanonicalMPS, Strategy, DEFAULT_STRATEGY
from ..operators import MPO
from ..truncate import simplify
from ..typing import Vector


def runge_kutta(
    H: MPO,
    t_span: Union[float, tuple[float, float], Vector],
    state: MPS,
    steps: int = 1000,
    strategy: Strategy = DEFAULT_STRATEGY,
    callback: Optional[Callable] = None,
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
    strategy : Strategy, default = DEFAULT_STRATEGY
        Truncation strategy for MPO and MPS algebra.
    callback : Optional[Callable[[float, MPS], Any]]
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
    for t in t_span:
        if t != last_t:
            idt = factor * (t - last_t)
            H_state = H.apply(state)
            state2 = simplify(state - (0.5 * idt) * H_state, strategy=strategy)
            H_state2 = H.apply(state2)
            state3 = simplify(state - (0.5 * idt) * H_state2, strategy=strategy)
            H_state3 = H.apply(state3)
            state4 = simplify(state - idt * H_state3, strategy=strategy)
            H_state4 = H.apply(state4)
            state = simplify(
                state - (idt / 6) * (H_state + 2 * H_state2 + 2 * H_state3 + H_state4),
                strategy=normalize_strategy,
            )
        if callback is not None:
            output.append(callback(t, state))
        last_t = t
    if callback is None:
        return state
    else:
        return output
