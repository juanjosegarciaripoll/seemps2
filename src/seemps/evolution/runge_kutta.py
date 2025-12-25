from __future__ import annotations
import numpy as np
from math import sqrt
from typing import Any
from ..state import MPS, Strategy, DEFAULT_STRATEGY, scprod, simplify
from ..operators import MPO
from .common import (
    ODECallback,
    ODEFunction,
    TimeSpan,
    ode_solver,
    make_generalized_MPO,
)


def runge_kutta(
    H: MPO | ODEFunction,
    time: TimeSpan,
    state: MPS,
    steps: int = 1000,
    strategy: Strategy = DEFAULT_STRATEGY,
    callback: ODECallback | None = None,
    itime: bool = False,
) -> MPS | list[Any]:
    r"""Solve a Schrodinger equation using a fourth order Runge-Kutta method.

    See :func:`seemps.evolution.euler` for a description of the
    missing function arguments and the function's output.

    Parameters
    ----------
    H : MPO | Callback[[float, MPS], MPS]
        Hamiltonian in MPO form, or a function that takes the time :math:`t` and
        a MPS and transforms it as in :math:`H(t)\psi`
    """
    GH: ODEFunction = make_generalized_MPO(H)

    def evolve_for_dt(
        t: float, state: MPS, factor: complex | float, dt: float, strategy: Strategy
    ) -> MPS:
        h = -factor * dt
        H_state = GH(t, state)
        state2 = simplify(state + (0.5 * h) * H_state, strategy=strategy)
        H_state2 = GH(t + dt / 2, state2)
        state3 = simplify(state + (0.5 * h) * H_state2, strategy=strategy)
        H_state3 = GH(t + dt / 2, state3)
        state4 = simplify(state + h * H_state3, strategy=strategy)
        H_state4 = GH(t + dt, state4)
        return simplify(
            state + (h / 6) * (H_state + 2 * H_state2 + 2 * H_state3 + H_state4),
            strategy=strategy,
        )

    return ode_solver(evolve_for_dt, time, state, steps, strategy, callback, itime)


# TODO: URGENT - Fix this integrator
def runge_kutta_fehlberg(
    H: MPO | ODEFunction,
    time: TimeSpan,
    state: MPS,
    steps: int = 1000,
    tolerance: float = 1e-8,
    strategy: Strategy = DEFAULT_STRATEGY,
    callback: ODECallback | None = None,
    itime: bool = False,
) -> MPS | list[Any]:
    r"""Solve a Schrodinger equation using a fourth order Runge-Kutta method.

    See :func:`seemps.evolution.euler` for a description of the
    function arguments that are not described below and the function's output.

    Parameters
    ----------
    H : MPO | Callback[[float, MPS], MPS]
        Hamiltonian in MPO form, or a function that takes the time :math:`t` and
        a MPS and transforms it as in :math:`H(t)\psi`
    tolerance : float, default = 1e-8
        Tolerance for determination of evolution step.
    """
    desired_dt: float = np.inf
    epsilon = np.finfo(np.float64).eps
    GH: ODEFunction = make_generalized_MPO(H)

    def evolve_for_dt(
        t: float,
        state: MPS,
        factor: complex | float,
        max_dt: float,
        normalize_strategy: Strategy,
    ) -> MPS:
        """Solve one evolution step with 4th-5th order Runge-Kutta-Fehlberg.
        We solve the equation dv/dt = factor * H.
        """
        nonlocal desired_dt
        left_dt = max_dt
        while left_dt > epsilon:
            while True:
                dt = min(desired_dt, max_dt)
                h = -factor * dt
                k1 = GH(t, state)
                state2 = simplify(state + (0.25 * h) * k1, strategy=strategy)
                k2 = GH(t + dt / 4, state2)
                state3 = simplify(
                    state + (3 * h / 32) * k1 + (9 * h / 32) * k2, strategy=strategy
                )
                k3 = GH(t + 3 * dt / 8, state3)
                state4 = simplify(
                    state
                    + (1932 * h / 2197) * k1
                    - (7200 * h / 2197) * k2
                    + (7296 * h / 2197) * k3,
                    strategy=strategy,
                )
                k4 = GH(t + 12 * dt / 13, state4)
                state5 = simplify(
                    state
                    + (439 / 216 * h) * k1
                    - (8 * h) * k2
                    + (3680 * h / 513) * k3
                    - (845 * h / 4104) * k4,
                    strategy=strategy,
                )
                k5 = GH(t + dt, state5)
                state6 = simplify(
                    state
                    - (8 * h / 27) * k1
                    + (2 * h) * k2
                    - (3544 * h / 2565) * k3
                    + (1859 * h / 4104) * k4
                    - (11 * h / 40) * k5,
                    strategy=strategy,
                )
                k6 = GH(t + dt / 2, state6)
                state_ord5 = simplify(
                    state
                    + (16 * h / 135) * k1
                    + (6656 * h / 12825) * k3
                    + (28561 * h / 56430) * k4
                    - (9 * h / 50) * k5
                    + (2 * h / 55) * k6,
                    strategy=normalize_strategy,
                )
                norm_ord5 = state_ord5.norm_squared()
                state_ord4 = simplify(
                    state
                    + (25 * h / 216) * k1
                    + (1408 * h / 2565) * k3
                    + (2197 * h / 4104) * k4
                    - (h / 5) * k5,
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
