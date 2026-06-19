from __future__ import annotations
from collections.abc import Sequence
from typing import Callable, Any, TypeAlias
import numpy as np
from ..typing import Real, Vector
from ..state import MPS, Strategy, DEFAULT_STRATEGY
from ..operators import MPO

ODEFunction: TypeAlias = Callable[[float, MPS], MPS]

GeneralizedMPO: TypeAlias = ODEFunction | MPO

ODECallback: TypeAlias = Callable[[float, MPS], Any]

TimeSpan: TypeAlias = float | tuple[Real, Real] | Sequence[Real] | Vector


def make_generalized_MPO(L: GeneralizedMPO) -> ODEFunction:
    the_MPO: MPO

    def mpo_derivative(t: float, state: MPS) -> MPS:
        return the_MPO.apply(state)

    if isinstance(L, MPO):
        the_MPO = L
        return mpo_derivative
    else:
        return L


def ode_solver(
    evolve_for_dt: Callable[[float, MPS, float, Strategy], MPS],
    time: TimeSpan,
    state: MPS,
    steps: int = 1000,
    strategy: Strategy = DEFAULT_STRATEGY,
    callback: ODECallback | None = None,
):
    r"""Abstract class for solving a linear ODE using MPOs and MPS.

    Parameters
    ----------
    evolve_for_dt: Callable
        Routine in charge of running each evolution step.
    time : float | tuple[float, float] | Vector
        Integration interval, or sequence of time steps.
    state : MPS
        Initial guess of the ground state.
    steps : int, default = 1000
        Integration steps, if not defined by `t_span`.
    strategy : Strategy, default = DEFAULT_STRATEGY
        Truncation strategy for MPO and MPS algebra.
    callback : Callable[[float, MPS], Any]
        A callable called after each iteration (defaults to None).
    """
    if isinstance(time, (int, float)):
        time = (0.0, float(time))
    if isinstance(time, tuple):
        t_span = np.linspace(
            float(time[0]), float(time[1]), steps + 1, dtype=np.float64
        )
    else:
        t_span = np.asarray(time, dtype=np.float64)

    output = []
    last_t = t_span[0]
    for t in t_span:
        if t != last_t:
            state = evolve_for_dt(t, state, float(t - last_t), strategy)
        if callback:
            output.append(callback(t, state))
        last_t = t
    return output if callback else state


__all__ = ["ode_solver", "ODECallback"]
