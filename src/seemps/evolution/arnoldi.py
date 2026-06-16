from __future__ import annotations
from typing import Any
from ..optimization.arnoldi import MPSArnoldiRepresentation
from ..state import MPS, Strategy, DEFAULT_STRATEGY
from ..operators import MPO
from .common import ode_solver, ODECallback, TimeSpan


def arnoldi(
    L: MPO,
    time: TimeSpan,
    state: MPS,
    steps: int = 1000,
    order: int = 6,
    strategy: Strategy = DEFAULT_STRATEGY,
    callback: ODECallback | None = None,
) -> MPS | list[Any]:
    r"""Solve ``d|state>/dt = L|state>`` using a variable order Arnoldi
    approximation to the exponential.

    See :func:`seemps.evolution.euler` for a description of the
    missing function arguments and the function's output.

    Parameters
    ----------
    order : int, default = 5
        Maximum order of the Arnoldi representation.
    """
    arnoldiL = None

    def evolve_for_dt(
        t: float,
        state: MPS,
        dt: float,
        strategy: Strategy,
    ) -> MPS:
        nonlocal arnoldiL
        if arnoldiL is None:
            arnoldiL = MPSArnoldiRepresentation(L, strategy)
        arnoldiL.build_Krylov_basis(state, order)
        return arnoldiL.exponential(dt)

    return ode_solver(evolve_for_dt, time, state, steps, strategy, callback)
