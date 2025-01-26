from __future__ import annotations
from typing import Callable
import numpy as np
from ..optimization.arnoldi import MPSArnoldiRepresentation
from ..typing import Vector
from ..state import MPS, Strategy, DEFAULT_STRATEGY
from ..operators import MPO


def arnoldi(
    H: MPO,
    t_span: float | tuple[float, float] | Vector,
    state: MPS,
    steps: int = 1000,
    order: int = 6,
    strategy: Strategy = DEFAULT_STRATEGY,
    callback: Callable | None = None,
    itime: bool = False,
):
    r"""Solve a Schrodinger equation using a variable order Arnoldi
    approximation to the exponential.

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
    order : int, default = 5
        Maximum order of the Arnoldi representation.
    strategy : Strategy, default = DEFAULT_STRATEGY
        Truncation strategy for MPO and MPS algebra.
    callback : Callable[[float, MPS], Any]
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
    arnoldiH = MPSArnoldiRepresentation(H, normalize_strategy)
    for t in t_span:
        if t != last_t:
            idt = factor * (t - last_t)
            arnoldiH.build_Krylov_basis(state, order)
            state = arnoldiH.exponential(-idt)
        if callback is not None:
            output.append(callback(t, state))
        last_t = t
    if callback is None:
        return state
    else:
        return output
