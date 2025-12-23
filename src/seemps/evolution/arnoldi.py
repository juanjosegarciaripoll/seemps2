from __future__ import annotations
from typing import Any
from ..optimization.arnoldi import MPSArnoldiRepresentation
from ..state import MPS, Strategy, DEFAULT_STRATEGY
from ..operators import MPO
from .common import ode_solver, ODECallback, TimeSpan


def arnoldi(
    H: MPO,
    time: TimeSpan,
    state: MPS,
    steps: int = 1000,
    order: int = 6,
    strategy: Strategy = DEFAULT_STRATEGY,
    callback: ODECallback | None = None,
    itime: bool = False,
) -> MPS | list[Any]:
    r"""Solve a Schrodinger equation using a variable order Arnoldi
    approximation to the exponential.

    See :func:`seemps.evolution.euler` for a description of the
    missing function arguments and the function's output.

    Parameters
    ----------
    order : int, default = 5
        Maximum order of the Arnoldi representation.
    """
    arnoldiH = None

    def evolve_for_dt(
        t: float,
        state: MPS,
        factor: complex | float,
        dt: float,
        normalize_strategy: Strategy,
    ) -> MPS:
        nonlocal arnoldiH
        if arnoldiH is None:
            arnoldiH = MPSArnoldiRepresentation(H, normalize_strategy)
        arnoldiH.build_Krylov_basis(state, order)
        idt = factor * dt
        return arnoldiH.exponential(-idt)

    return ode_solver(evolve_for_dt, time, state, steps, strategy, callback, itime)
