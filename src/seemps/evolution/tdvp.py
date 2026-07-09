from __future__ import annotations

import numpy as np
from scipy.linalg import expm as _expm
from typing import Any
from seemps.state import MPS, CanonicalMPS, Strategy, DEFAULT_STRATEGY
from seemps.cython import _contract_last_and_first
from seemps.operators import MPO
from seemps.operators.quadratic import QuadraticForm
from seemps.evolution.common import ode_solver, ODECallback, TimeSpan


def _expmv(
    A: Any, v: np.ndarray, tau: complex, tol: float = 1e-12, m_max: int = 30
) -> np.ndarray:
    """exp(tau*A) @ v via Arnoldi iteration with Rayleigh-quotient shift
    and cheap leading-order early convergence checks"""
    flat = v.ravel().astype(complex, copy=False)
    norm_v = float(np.linalg.norm(flat))
    if norm_v == 0.0:
        return v
    n = len(flat)
    Q = np.zeros((n, m_max + 1), dtype=complex)
    H = np.zeros((m_max + 1, m_max), dtype=complex)
    Q[:, 0] = flat / norm_v
    mu = None
    tol_v = tol * norm_v
    abs_tau = abs(tau)
    abs_tau_pow = inv_fact = prod_h = 1.0
    m = 0
    e1p = None
    for j in range(m_max):
        w = (A @ Q[:, j]).astype(complex, copy=False)
        h = Q[:, : j + 1].conj().T @ w
        H[: j + 1, j] = h
        w -= Q[:, : j + 1] @ h
        if mu is None:
            mu = H[0, 0]
        hn = float(np.linalg.norm(w))
        H[j + 1, j] = hn
        m = j + 1
        # Leading-order Saad bound to skip expm when provably unconverged
        if hn * abs_tau_pow * inv_fact * prod_h > tol_v and hn >= 1e-14:
            prod_h *= hn
            abs_tau_pow *= abs_tau
            inv_fact /= m
            Q[:, j + 1] = w / hn
            continue
        # Near convergence: compute exact Saad error via expm
        Hm = H[:m, :m].copy()
        Hm.flat[:: m + 1] -= mu
        e1p = _expm(tau * Hm)[:, 0]
        if hn * abs(e1p[m - 1]) <= tol_v or hn < 1e-14:
            break
        prod_h *= hn
        abs_tau_pow *= abs_tau
        inv_fact /= m
        Q[:, j + 1] = w / hn
    if mu is None:
        raise RuntimeError("Arnoldi iteration produced no vectors")
    if e1p is None:
        Hm = H[:m, :m].copy()
        Hm.flat[:: m + 1] -= mu
        e1p = _expm(tau * Hm)[:, 0]
    return (np.exp(tau * mu) * norm_v * (Q[:, :m] @ e1p)).reshape(v.shape)


def tdvp_step(
    L: MPO, state: MPS, dt: float | complex, strategy: Strategy = DEFAULT_STRATEGY
) -> CanonicalMPS:
    if not isinstance(state, CanonicalMPS):
        state = CanonicalMPS(state, center=0, strategy=strategy)

    QF = QuadraticForm(L, state, start=0)

    def _evolve(Op: Any, tensor: np.ndarray, factor: float | complex) -> np.ndarray:
        v = _expmv(Op, tensor.ravel(), tau=factor)
        return v.reshape(tensor.shape)

    # Sweep Right
    for i in range(L.size - 1):
        A2 = _contract_last_and_first(QF.state[i], QF.state[i + 1])
        QF.update_2site_right(
            _evolve(QF.two_site_operator(i), A2, 0.5 * dt), i, strategy
        )
        if i < L.size - 2:
            QF.state[i + 1] = _evolve(
                QF.one_site_operator(i + 1), QF.state[i + 1], -0.5 * dt
            )

    # Sweep Left
    for i in range(L.size - 2, -1, -1):
        A2 = _contract_last_and_first(QF.state[i], QF.state[i + 1])
        QF.update_2site_left(
            _evolve(QF.two_site_operator(i), A2, 0.5 * dt), i, strategy
        )
        if i > 0:
            QF.state[i] = _evolve(QF.one_site_operator(i), QF.state[i], -0.5 * dt)

    return QF.state


def tdvp(
    L: MPO,
    time: TimeSpan,
    state: MPS,
    steps: int = 1000,
    strategy: Strategy = DEFAULT_STRATEGY,
    callback: ODECallback | None = None,
):
    r"""Solve ``d|state>/dt = L|state>`` using the Time Dependent Variational Principle
    (TDVP) algorithm.

    Parameters
    ----------
    L : MPO
        Linear operator in MPO form.
    time : Real | tuple[Real, Real] | Sequence[Real]
        Integration interval, or sequence of time steps.
    state : MPS
        Initial guess of the ground state.
    steps : int, default = 1000
        Integration steps, if not defined by `t_span`.
    strategy : Strategy, default = DEFAULT_STRATEGY
        Truncation strategy for MPO and MPS algebra.
    callback : Callable[[float, MPS], Any] | None
        A callable called after each iteration (defaults to None).
    Returns
    -------
    result : MPS | list[Any]
        Final state after evolution or values collected by callback
    """

    def evolve_for_dt(t: float, state: MPS, dt: float, strategy: Strategy) -> MPS:
        return tdvp_step(L, state, dt, strategy=strategy)

    return ode_solver(evolve_for_dt, time, state, steps, strategy, callback)
