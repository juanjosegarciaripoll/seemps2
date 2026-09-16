from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import lru_cache
from typing import Any, TypeAlias

import numpy as np

from ..operators import MPO, MPOSum
from ..operators.projectors import identity_mpo
from ..operators.simplify_mpo import simplify_mpo
from ..solve import dmrg_solve
from ..state import (
    DEFAULT_STRATEGY,
    MPS,
    CanonicalMPS,
    MPSSum,
    Strategy,
    simplify,
)
from .common import ODECallback, TimeSpan, ode_solver

NonlinearTerm: TypeAlias = Callable[[float, MPS], MPO]

# Butcher tableau of a collocation method: nodes c, matrix A, weights b, A^{-1}.
Tableau = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]


def _tableau(c: np.ndarray) -> Tableau:
    """Nodes c, Butcher matrix A, weights b and A^{-1}."""
    k = np.arange(len(c))[:, np.newaxis]
    V = c**k
    A = np.linalg.solve(V, c ** (k + 1) / (k + 1)).T
    b = np.linalg.solve(V, 1.0 / (k + 1)).ravel()
    return c, A, b, np.linalg.inv(A)


def _decoupled_weights(A: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Eigenvalues of A to decouple stages in a linear problem,
    and the weights recombining the resulting stage solutions."""
    vals, vecs = np.linalg.eig(A)
    coeffs = np.linalg.solve(vecs, np.ones(len(vals)))
    return vals, coeffs * (b @ vecs) / vals


@lru_cache
def _gl_tableau(stages: int) -> Tableau:
    """Tableau of the Gauss-Legendre method."""
    x, _ = np.polynomial.legendre.leggauss(stages)
    return _tableau((x + 1) / 2)


def _solve_stage(
    A: MPO, guess: CanonicalMPS, rhs: MPS, eps: float, strategy: Strategy
) -> CanonicalMPS:
    return dmrg_solve(
        A,
        rhs,
        guess=guess,
        strategy=strategy,
        rtol=eps,
        compute_residuals=False,
    )[0]


def _combination(
    state: CanonicalMPS,
    w: np.ndarray,
    Y: Sequence[MPS],
    guess: MPS,
    strategy: Strategy,
) -> CanonicalMPS:
    """Simplify ``state + sum_j w_j (Y_j - state)`` in a single pass."""
    return simplify(
        MPSSum([1.0 - sum(w), *w], [state, *Y]), guess=guess, strategy=strategy
    )


def _decoupled_step(
    L: MPO,
    state: CanonicalMPS,
    dt: complex,
    vals: np.ndarray,
    g: np.ndarray,
    eps: float,
    strategy: Strategy,
) -> CanonicalMPS:
    I = identity_mpo(L.physical_dimensions())
    ops = [MPOSum([I, L], [1.0, -dt * v]).join() for v in vals]
    # `_solve_stage` overwrites its guess in place.
    xs = [_solve_stage(op, state.copy(), state, eps, strategy) for op in ops]
    return _combination(state, g, xs, state, strategy)


def _converged(change: float, prev: float, eps: float, sweep: int, min_sweep: int = 2) -> bool:
    """Converged, or no longer improving after a few sweeps."""
    return change <= eps or (change >= prev and sweep >= min_sweep)


def _predictor(
    L: MPO,
    N: NonlinearTerm,
    t: float,
    state: CanonicalMPS,
    dt: complex,
    c: np.ndarray,
    strategy: Strategy,
) -> list[CanonicalMPS]:
    """Euler seed Y_s = state + dt*cs*(L+N)|state>."""
    k0 = simplify(
        MPSSum([1.0, 1.0], [L.apply(state), N(t, state).apply(state)]),
        strategy=strategy,
    )
    return [
        simplify(MPSSum([1.0, dt * cs], [state, k0]), guess=state, strategy=strategy)
        for cs in c
    ]


def _nonlinear_step(
    L: MPO,
    N: NonlinearTerm,
    t: float,
    state: CanonicalMPS,
    dt: complex,
    tableau: Tableau,
    eps: float,
    strategy: Strategy,
) -> CanonicalMPS:
    """Gauss-Seidel over the stage values Y_s."""
    c, A, b, Ainv = tableau
    norm = state.norm()
    Y = _predictor(L, N, t, state, dt, c, strategy)
    I = identity_mpo(L.physical_dimensions())
    prev = np.inf
    for sweep in range(strategy.get_max_sweeps()):
        change = 0.0
        for s in range(len(c)):
            op = simplify_mpo(
                MPOSum([L, N(t + c[s] * dt, Y[s])], [-dt * A[s, s]] * 2), strategy
            )
            op = MPOSum([I, op]).join()
            w =-A[s, s] * Ainv[s]  # rhs_s = state + dt*sum_{j!=s} A_sj k_j, via Y
            w[s] += 1.0
            rhs = _combination(state, w, Y, Y[s], strategy)
            new = _solve_stage(op, Y[s].copy(), rhs, eps, strategy)
            change = max(change, (new - Y[s]).norm() / norm)
            Y[s] = new
        if _converged(change, prev, eps, sweep):
            break
        prev = change
    if abs(c[-1] - 1.0) < 1e-14:
        return CanonicalMPS(Y[-1], center=0, strategy=strategy)
    return _combination(state, b @ Ainv, Y, state, strategy)


def gausslegendre_step(
    L: MPO,
    t: float,
    state: MPS,
    dt: complex,
    stages: int = 2,
    strategy: Strategy = DEFAULT_STRATEGY,
    rtol: float = 1e-5,
    N: NonlinearTerm | None = None,
) -> CanonicalMPS:
    """Advance one implicit Gauss-Legendre step, of order 2*`stages`.

    Parameters
    ----------
    L : MPO
        Constant part of the generator.
    t : float
        Time at the beginning of the step.
    state : MPS
        State at the beginning of the step.
    dt : complex
        Time step.
    stages : int, default = 2
        Number of collocation stages. `stages = 1` is the implicit midpoint rule.
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
        raise ValueError("gausslegendre requires at least one stage")
    strategy = strategy.replace(normalize=False)
    state = CanonicalMPS(state, center=0, strategy=strategy)
    tableau = _gl_tableau(stages)
    if N is None:
        _, A, b, _ = tableau
        return _decoupled_step(L, state, dt, *_decoupled_weights(A, b), rtol, strategy)
    return _nonlinear_step(L, N, t, state, dt, tableau, rtol, strategy)


def gausslegendre(
    L: MPO,
    time: TimeSpan,
    state: MPS,
    steps: int = 1000,
    stages: int = 2,
    strategy: Strategy = DEFAULT_STRATEGY,
    callback: ODECallback | None = None,
    rtol: float = 1e-5,
    N: NonlinearTerm | None = None,
) -> MPS | list[Any]:
    r"""Solve ``d|state>/dt = (L + N(t, state))|state>`` using an implicit
    Gauss-Legendre method of order 2*`stages`.

    Gauss-Legendre collocation gives the highest order attainable with a given
    number of stages, is symplectic and A-stable.
    It accepts a state-dependent generator: `N` is the nonlinear part of the
    equation, supplied as a function returning the MPO that acts on the state
    at a given time.

    See :func:`seemps.evolution.euler` for a description of the missing
    function arguments and the function's output.

    Parameters
    ----------
    L : MPO
        Constant part of the generator.
    stages : int, default = 2
        Number of collocation stages, giving order `2*stages`. `stages = 1` is
        the implicit midpoint rule.
    N : NonlinearTerm | None
        State-dependent part of the generator, ``N(t, state) -> MPO``.
    rtol : float, default = 1e-5
        Relative tolerance of the linear solves at each stage.
    """

    def evolve_for_dt(
        t: float,
        current_state: MPS,
        dt: float,
        current_strategy: Strategy,
    ) -> MPS:
        return gausslegendre_step(
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
