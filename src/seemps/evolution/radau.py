from __future__ import annotations
from typing import Any, TypeVar
import numpy as np
from ..operators import MPO
from ..state import DEFAULT_STRATEGY, MPS, Strategy, simplify
from ..solve import dmrg_solve
from ..operators.projectors import identity_mpo
from ..operators.simplify_mpo import simplify_mpo
from .common import ode_solver, ODECallback, TimeSpan


# Map number of stages to Butcher matrix A
A = {
    3: np.array(
        [
            [0.196815477223660, -0.065535425850198, 0.023770974348220],
            [0.394424314739087, 0.292073411665228, -0.041548752125998],
            [0.376403062700467, 0.512485826188421, 0.111111111111111],
        ]
    ),
    5: np.array(
        [
            [
                0.072998864318,
                -0.026735331108,
                0.018676929764,
                -0.012879106093,
                0.005042839234,
            ],
            [
                0.153775231479,
                0.146214867847,
                -0.036444568905,
                0.021233063119,
                -0.007935579903,
            ],
            [
                0.140063045685,
                0.298967129491,
                0.167585070135,
                -0.033969101687,
                0.010944288744,
            ],
            [
                0.144894308110,
                0.276500068760,
                0.325797922910,
                0.128756753255,
                -0.015708917379,
            ],
            [
                0.143713560791,
                0.281356015149,
                0.311826522976,
                0.223103901084,
                0.040000000000,
            ],
        ]
    ),
}

# Map number of stages to Runge-Kutta weights b
b = {
    3: np.array([0.376403062700467, 0.512485826188421, 0.111111111111111]),
    5: np.array(
        [0.143713560791, 0.281356015149, 0.311826522976, 0.223103901084, 0.040000000000]
    ),
}

StateOrOperator = TypeVar("StateOrOperator", MPO, MPS)


def _prepend_core(core: np.ndarray, L: StateOrOperator) -> StateOrOperator:
    data = [core] + L._data
    return type(L)(data)


def radau_step(
    L: MPO,
    v: MPS,
    dt: float | complex,
    stages: int = 3,
    inv_tol: float | None = None,
    strategy: Strategy = DEFAULT_STRATEGY,
) -> MPS:
    # Number of steps
    m = len(b[stages])

    # Extended identity, operator L and rhs vector
    dimensions = [site.shape[1] for site in L]
    Im = _prepend_core(np.eye(m).reshape(1, m, m, 1), identity_mpo(dimensions))
    Lm = _prepend_core(A[stages].reshape(1, m, m, 1), L)
    rhs = _prepend_core(np.ones((1, m, 1)), simplify(L @ v, strategy))
    Dm = simplify_mpo((Im - dt * Lm).join(), strategy)

    # Solve linear system
    if inv_tol is None:
        inv_tol = strategy.get_simplification_tolerance()
    Km, _ = dmrg_solve(Dm, rhs, strategy=strategy, rtol=inv_tol)

    # Sum over step weights b
    # np.einsum('b,abc,cde->ade', b, KM[0], KM[1])
    K_core_0 = np.tensordot(b[stages], Km[0], axes=([0], [1]))
    K_core_0 = np.tensordot(K_core_0, Km[1], axes=([1], [0]))

    # Update solution
    update = MPS([K_core_0] + Km._data[2:])
    solution = simplify(v + dt * update, strategy)

    return solution


def radau(
    H: MPO,
    time: TimeSpan,
    state: MPS,
    steps: int = 1000,
    stages: int = 3,
    inv_tol: float = 1e-7,
    strategy: Strategy = DEFAULT_STRATEGY,
    callback: ODECallback | None = None,
    itime: bool = False,
) -> MPS | list[Any]:
    r"""Solve a Schrödinger equation using an implicit Radau IIA method with either
    3 or 5 stages (order 5 or 9, respectively).

    See :func:`seemps.evolution.euler` for a description of the
    missing function arguments and the function's output.

    Parameters
    ----------
    stages : int, default = 3
        Number of Radau IIA stages (3 or 5).
    inv_tol : float, default = 1e-7
        Tolerance for the GMRES solver.
    """

    def evolve_for_dt(
        t: float,
        state: MPS,
        factor: complex | float,
        dt: float,
        normalize_strategy: Strategy,
    ) -> MPS:
        idt = factor * dt
        return radau_step(
            L=H,
            v=state,
            dt=-idt,
            inv_tol=inv_tol,
            strategy=normalize_strategy,
            stages=stages,
        )

    return ode_solver(evolve_for_dt, time, state, steps, strategy, callback, itime)
