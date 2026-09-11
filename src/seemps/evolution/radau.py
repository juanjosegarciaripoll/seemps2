from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np
from scipy.special import roots_jacobi

from ..operators import MPO
from ..state import (
    DEFAULT_STRATEGY,
    MPS,
    CanonicalMPS,
    Strategy,
)
from .common import ODECallback, TimeSpan, ode_solver
from .gausslegendre import (
    NonlinearTerm,
    Tableau,
    _decoupled_step,
    _decoupled_weights,
    _nonlinear_step,
    _tableau,
)


@lru_cache
def _radau_tableau(stages: int) -> Tableau:
    """Tableau of the Radau IIA method."""
    if stages == 1:
        return _tableau(np.array([1.0]))
    x, _ = roots_jacobi(stages - 1, 1, 0)
    return _tableau(np.sort(np.append((x + 1) / 2, 1.0)))


def radau_step(
    L: MPO,
    t: float,
    state: MPS,
    dt: complex,
    stages: int = 3,
    strategy: Strategy = DEFAULT_STRATEGY,
    rtol: float = 1e-5,
    N: NonlinearTerm | None = None,
) -> CanonicalMPS:
    """One implicit Radau IIA step, order `2*stages - 1`.

    Parameters
    ----------
    L : MPO
        Constant part of the generator.
    t : float
        Time at the beginning of the step.
    state : MPS
        State at the beginning of the step.
    dt : float | complex
        Time step.
    stages : int, default = 3
        Number of Radau IIA stages, giving order 2*`stages` - 1. `stages = 1`
        is the backward Euler method.
    strategy : Strategy, default = DEFAULT_STRATEGY
        Truncation strategy for MPO and MPS algebra.
    rtol : float, default = 1e-5
        Relative tolerance of the stage solves, as in :func:`seemps.solve.dmrg_solve`.
    N : NonlinearTerm | None
        State-dependent part of the generator. When it is `None` the stages
        decouple exactly and the step costs one linear solve per stage.

    Returns
    -------
    CanonicalMPS
        State at the end of the step.
    """
    if stages < 1:
        raise ValueError("radau requires at least one stage")
    strategy = strategy.replace(normalize=False)
    state = CanonicalMPS(state, center=0, strategy=strategy)
    tableau = _radau_tableau(stages)
    if N is None:
        _, A, b, _ = tableau
        return _decoupled_step(L, state, dt, *_decoupled_weights(A, b), rtol, strategy)
    return _nonlinear_step(L, N, t, state, dt, tableau, rtol, strategy)


def radau(
    L: MPO,
    time: TimeSpan,
    state: MPS,
    steps: int = 1000,
    stages: int = 3,
    strategy: Strategy = DEFAULT_STRATEGY,
    callback: ODECallback | None = None,
    rtol: float = 1e-5,
    N: NonlinearTerm | None = None,
) -> MPS | list[Any]:
    r"""Solve ``d|state>/dt = (L + N(t, state))|state>`` using an implicit
    Radau IIA method of order (2 * `stages` - 1).

    Radau IIA is stiffly accurate and L-stable, which makes it suited to
    equations with strongly dissipative parts.
    It accepts a state-dependent generator: `N` is the nonlinear part of the
    equation, supplied as a function returning the MPO that acts on the state
    at a given time.

    See :func:`seemps.evolution.euler` for a description of the missing
    function arguments and the function's output.

    Parameters
    ----------
    L : MPO
        Constant part of the generator.
    stages : int, default = 3
        Number of Radau IIA stages, giving order 2*`stages` - 1. `stages = 1`
        is the backward Euler method.
    N : NonlinearTerm | None
        State-dependent part of the generator, ``N(t, state) -> MPO``. When it
        is `None` (the default) the equation is linear, the stages decouple
        through the eigenbasis of the Butcher matrix, and each step costs one
        linear solve per stage. Otherwise the stage values are found by a
        Gauss-Seidel iteration, whose number of sweeps is bounded by
        ``strategy.get_max_sweeps()``.
    rtol : float, default = 1e-5
        Relative tolerance of the linear solves at each stage.
    """

    def evolve_for_dt(
        t: float,
        current_state: MPS,
        dt: float,
        current_strategy: Strategy,
    ) -> MPS:
        return radau_step(
            L,
            t,
            current_state,
            dt,
            stages=stages,
            strategy=current_strategy,
            rtol=rtol,
            N=N,
        )

    return ode_solver(evolve_for_dt, time, state, steps, strategy, callback)
