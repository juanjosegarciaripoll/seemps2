from __future__ import annotations
import numpy as np
from typing import Union, Optional, Callable
from ..state import MPS, Strategy, DEFAULT_STRATEGY
from ..expectation import scprod
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


def runge_kutta_fehlberg(
    H: MPO,
    t_span: Union[float, tuple[float, float], Vector],
    state: MPS,
    steps: int = 1000,
    tolerance: float = 1e-8,
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
    tolerance : float, default = 1e-8
        Tolerance for determination of evolution step.
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
        factor = -1
        normalize_strategy = strategy.replace(normalize=True)
    else:
        factor = -1j
        normalize_strategy = strategy
    last_t = t_span[0]
    output = []

    def _rk45_one_step(
        state: MPS, desired_dt: float, max_dt: float
    ) -> tuple[MPS, float, float]:
        """Solve one evolution step with 4th-5th order Runge-Kutta-Fehlberg.
        We solve the equation dv/dt = factor * H.
        """
        while True:
            dt = min(desired_dt, max_dt)
            idt = factor * dt
            k1 = H.apply(state)
            state2 = simplify(state + 0.25 * idt * k1, strategy=strategy)
            k2 = H.apply(state2)
            state3 = simplify(
                state + (3 / 32) * idt * k1 + (9 / 32) * idt * k2, strategy=strategy
            )
            k3 = H.apply(state3)
            state4 = simplify(
                state
                + (1932 / 2197) * idt * k1
                - (7200 / 2197) * idt * k2
                + (7296 / 2197) * idt * k3,
                strategy=strategy,
            )
            k4 = H.apply(state4)
            state5 = simplify(
                state
                + (439 / 216) * idt * k1
                - 8 * idt * k2
                + (3680 / 513) * idt * k3
                - (845 / 4104) * idt * k4,
                strategy=strategy,
            )
            k5 = H.apply(state5)
            state6 = simplify(
                state
                - (8 / 27) * idt * k1
                + 2 * idt * k2
                - (3544 / 2565) * idt * k3
                + (1859 / 4104) * idt * k4
                - (11 / 40) * idt * k5,
                strategy=strategy,
            )
            k6 = -1 * H.apply(state6)
            state_ord5 = simplify(
                state
                + idt
                * (
                    (16 / 135) * k1
                    + (6656 / 12825) * k3
                    + (28561 / 56430) * k4
                    - (9 / 50) * k5
                    + (2 / 55) * k6
                ),
                strategy=normalize_strategy,
            )
            norm_ord5 = state_ord5.norm_squared()
            state_ord4 = simplify(
                state
                + idt
                * (
                    (25 / 216) * k1
                    + (1408 / 2565) * k3
                    + (2197 / 4104) * k4
                    - (1 / 5) * k5
                ),
                strategy=normalize_strategy,
            )
            norm_ord4 = state_ord5.norm_squared()
            delta = abs(
                2
                * (
                    1
                    - scprod(state_ord5, state_ord4).real
                    / np.sqrt(norm_ord5 * norm_ord4)
                )
            )
            if delta > 0:
                desired_dt = 0.9 * dt * (tolerance / delta) ** 0.2
            if delta <= tolerance:
                return state_ord5, dt, desired_dt

    dt = np.inf
    epsilon = np.finfo(np.float64).eps
    for t in t_span:
        while last_t + epsilon < t:
            state, actual_dt, dt = _rk45_one_step(
                state, min(dt, t - last_t), t - last_t
            )
            last_t += actual_dt
        if callback is not None:
            output.append(callback(t, state))
        last_t = t
    if callback is None:
        return state
    else:
        return output
