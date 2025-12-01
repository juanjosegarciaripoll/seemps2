from __future__ import annotations
import numpy as np
from math import sqrt
from typing import Any
from ..state import MPS, Strategy, DEFAULT_STRATEGY, scprod
from ..operators import MPO
from ..truncate import simplify
from .common import ODECallback, TimeSpan, ode_solver


def runge_kutta(
    H: MPO,
    time: TimeSpan,
    state: MPS,
    steps: int = 1000,
    strategy: Strategy = DEFAULT_STRATEGY,
    callback: ODECallback | None = None,
    itime: bool = False,
) -> MPS | list[Any]:
    r"""Solve a Schrodinger equation using a fourth order Runge-Kutta method.

    See :function:`seemps.evolution.euler` for a description of the
    missing function arguments and the function's output.
    """

    def evolve_for_dt(
        state: MPS, factor: complex | float, dt: float, strategy: Strategy
    ) -> MPS:
        idt = factor * dt
        H_state = H.apply(state)
        state2 = simplify(state - (0.5 * idt) * H_state, strategy=strategy)
        H_state2 = H.apply(state2)
        state3 = simplify(state - (0.5 * idt) * H_state2, strategy=strategy)
        H_state3 = H.apply(state3)
        state4 = simplify(state - idt * H_state3, strategy=strategy)
        H_state4 = H.apply(state4)
        return simplify(
            state - (idt / 6) * (H_state + 2 * H_state2 + 2 * H_state3 + H_state4),
            strategy=strategy,
        )

    return ode_solver(evolve_for_dt, time, state, steps, strategy, callback, itime)


# TODO: URGENT - Fix this integrator
def runge_kutta_fehlberg(
    H: MPO,
    time: TimeSpan,
    state: MPS,
    steps: int = 1000,
    tolerance: float = 1e-8,
    strategy: Strategy = DEFAULT_STRATEGY,
    callback: ODECallback | None = None,
    itime: bool = False,
) -> MPS | list[Any]:
    r"""Solve a Schrodinger equation using a fourth order Runge-Kutta method.

    See :function:`seemps.evolution.euler` for a description of the
    function arguments that are not described below and the function's output.

    Parameters
    ----------
    tolerance : float, default = 1e-8
        Tolerance for determination of evolution step.
    """
    desired_dt: float = 0.0
    epsilon = np.finfo(np.float64).eps

    def evolve_for_dt(
        state: MPS, factor: complex | float, max_dt: float, normalize_strategy: Strategy
    ) -> MPS:
        """Solve one evolution step with 4th-5th order Runge-Kutta-Fehlberg.
        We solve the equation dv/dt = factor * H.
        """
        nonlocal desired_dt
        left_dt = max_dt
        while left_dt > epsilon:
            while True:
                if desired_dt:
                    dt = min(desired_dt, max_dt)
                else:
                    dt = max_dt
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
                k6 = H.apply(state6)
                state_ord5 = simplify(
                    state
                    - idt
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
                    - idt
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
                        / sqrt(norm_ord5 * norm_ord4)
                    )
                )
                if delta > 0:
                    desired_dt = 0.9 * dt * (tolerance / delta) ** 0.2
                if delta <= tolerance:
                    state = state_ord5
                    left_dt -= dt
                    break
        return state

    return ode_solver(evolve_for_dt, time, state, steps, strategy, callback, itime)

