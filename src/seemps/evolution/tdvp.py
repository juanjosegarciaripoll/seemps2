from __future__ import annotations

import numpy as np
from scipy.sparse.linalg import LinearOperator, expm_multiply
from seemps.state import MPS, CanonicalMPS, Strategy, DEFAULT_STRATEGY
from seemps.cython import _contract_last_and_first
from seemps.operators import MPO
from seemps.operators.quadratic import QuadraticForm
from seemps.evolution.common import ode_solver, ODECallback, TimeSpan


def _evolve(
    operator: LinearOperator,
    tensor: np.ndarray,
    factor: float | complex,
    normalize: bool = False,
) -> np.ndarray:
    """Apply time evolution operator to tensor."""
    shape = tensor.shape
    operator_trace = getattr(operator, "trace", None)
    traceA = None if operator_trace is None else factor * operator_trace()
    v = expm_multiply(factor * operator, tensor.ravel(), traceA=traceA)
    if normalize:
        v = v / np.linalg.norm(v)

    return v.reshape(shape)


def tdvp_step(
    L: MPO, state: MPS, dt: float | complex, strategy: Strategy = DEFAULT_STRATEGY
) -> CanonicalMPS:
    if not isinstance(state, CanonicalMPS):
        state = CanonicalMPS(state, center=0, strategy=strategy)

    QF = QuadraticForm(L, state, start=0)
    normalize = strategy.get_normalize_flag()

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
