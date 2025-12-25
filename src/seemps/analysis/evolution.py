from __future__ import annotations
from typing import Any, Callable
from math import sqrt
import numpy as np
import dataclasses
from ..optimization.descent import DESCENT_STRATEGY
from ..state import MPS, CanonicalMPS, MPSSum, Strategy, scprod, simplify
from ..operators import MPO, MPOList, MPOSum


@dataclasses.dataclass
class EvolutionResults:
    """Results from ground state search using imaginary time evolution.

    Parameters
    ----------
    state : MPS | np.ndarray
        The estimate for the ground state.
    energy : float
        Estimate for the ground state energy.
    trajectory : Vector | None
        Vector of computed energies in the evolution trajectory.
    Δβ : float | list[float] | None
        Steps size or steps sizes for each iteration.
    β : np.ndarray
        Imaginary time evolution path.
    """

    state: MPS | np.ndarray
    energy: float
    trajectory: list[float] = dataclasses.field(default_factory=list)
    Δβ: float | list[float] | None = None
    β: list[float] = dataclasses.field(default_factory=list)


def euler(
    H: MPO | MPOList | MPOSum,
    state: MPS,
    Δβ: float = 0.01,
    maxiter: int = 1000,
    strategy: Strategy = DESCENT_STRATEGY,
    callback: Callable[[MPS, EvolutionResults], Any] | None = None,
) -> EvolutionResults:
    """Euler method for arrays.

    Parameters
    ----------
    H : MPO | MPOList | MPOSum
        Hamiltonian in MPO form.
    state : MPS
        Initial guess of the ground state.
    Δβ : float
        Step size (defaults to 0.01).
    maxiter : int
        Maximum number of iterations (defaults to 1000)
    strategy : Strategy | None
        Truncation strategy when applying MPO. Defaults to DESCENT_STRATEGY, thereby
        using whatever strategy the MPO has defined.
    callback : Callable[[MPS, EvolutionResults], Any] | None
        A callable called after each iteration (defaults to None).

    Returns
    -------
    EvolutionResults
        Results from the evolution. See :class:`EvolutionResults`.
    """
    normalization_strategy = strategy.replace(normalize=True)
    state = CanonicalMPS(state, normalize=True)
    results = EvolutionResults(state=state, energy=np.inf, trajectory=[], Δβ=Δβ, β=[0])
    for i in range(maxiter + 1):
        if i > 0:
            H_state = H.apply(state)
            state = simplify(state - Δβ * H_state, strategy=normalization_strategy)
            results.β.append(results.β[-1] + Δβ)
        E = H.expectation(state).real
        if callback is not None:
            callback(state, results)
        results.trajectory.append(E)
        if E < results.energy:
            results.energy, results.state = E, state
    return results


def improved_euler(
    H: MPO | MPOList | MPOSum,
    state: MPS,
    Δβ: float = 0.01,
    maxiter: int = 1000,
    strategy: Strategy = DESCENT_STRATEGY,
    callback: Callable[[MPS, EvolutionResults], Any] | None = None,
):
    """Improved Euler method for arrays. See `euler` for a description of
    parameters and results.
    """
    normalization_strategy = strategy.replace(normalize=True)
    state = CanonicalMPS(state, normalize=True)
    results = EvolutionResults(state=state, energy=np.inf, trajectory=[], Δβ=Δβ, β=[0])
    for i in range(maxiter):
        if i > 0:
            H_state = H.apply(state)
            k = simplify(2 * state - Δβ * H_state, strategy=strategy)
            Hk = H.apply(k)
            state = simplify(state - 0.5 * Δβ * Hk, strategy=normalization_strategy)
            results.β.append(results.β[-1] + Δβ)
        E = H.expectation(state).real
        if callback is not None:
            callback(state, results)
        results.trajectory.append(E)
        if E < results.energy:
            results.energy, results.state = E, state
    return results


def runge_kutta(
    H: MPO | MPOList | MPOSum,
    state: MPS,
    Δβ: float = 0.01,
    maxiter: int = 1000,
    strategy: Strategy = DESCENT_STRATEGY,
    callback: Callable[[MPS, EvolutionResults], Any] | None = None,
) -> EvolutionResults:
    """Runge-Kutta method for arrays. See `euler` for a description of
    parametrs and results.
    """
    normalization_strategy = strategy.replace(normalize=True)
    state = CanonicalMPS(state, normalize=True)
    results = EvolutionResults(state=state, energy=np.inf, trajectory=[], Δβ=Δβ, β=[0])
    for i in range(maxiter):
        if i > 0:
            H_state = H.apply(state)
            state2 = simplify(state - 0.5 * Δβ * H_state, strategy=strategy)
            H_state2 = H.apply(state2)
            state3 = simplify(state - 0.5 * Δβ * H_state2, strategy=strategy)
            H_state3 = H.apply(state3)
            state4 = simplify(state - Δβ * H_state3, strategy=strategy)
            H_state4 = H.apply(state4)
            state = simplify(
                state - Δβ / 6 * (H_state + 2 * H_state2 + 2 * H_state3 + H_state4),
                strategy=normalization_strategy,
            )
            results.β.append(results.β[-1] + Δβ)
        E = H.expectation(state).real
        if callback is not None:
            callback(state, results)
        results.trajectory.append(E)
        if E < results.energy:
            results.energy, results.state = E, state
    return results


def runge_kutta_fehlberg(
    H: MPO | MPOList | MPOSum,
    state: MPS,
    Δβ: float = 0.01,
    maxiter: int = 1000,
    tol_rk: float = 1e-8,
    tol_residual: float = 1e-8,
    strategy: Strategy = DESCENT_STRATEGY,
    callback: Callable[[MPS, EvolutionResults], Any] | None = None,
) -> EvolutionResults:
    """Runge-Kutta method for arrays.

    Parameters
    ----------
    H : MPO | MPOList | MPOSum
        Hamiltonian in MPO form.
    state : MPS
        Initial guess of the ground state.
    Δβ : float
        Step size (defaults to 0.01).
    maxiter : int
        Maximum number of iterations (defaults to 1000)
    tol_rk : float
        Maximum error difference between 4-th and 5-th order methods. It is used
        to adapt the time step: smaller errors may lead to shorter time steps
        and slower computations; large errors may lead to significant deviations
        from target (dfaults to 1e-8).
    tol_residual : float
        Tolerance in relative residual at which algorithm stops (defaults to 1e-10).
    strategy : Strategy | None
        Truncation strategy when applying MPO. Defaults to DESCENT_STRATEGY, thereby
        using whatever strategy the MPO has defined.
    callback : Callable[[MPS, EvolutionResults], Any] | None
        A callable called after each iteration (defaults to None).

    Returns
    -------
    EvolutionResults
        Results from the evolution. See :class:`EvolutionResults`.
    """
    normalization_strategy = strategy.replace(normalize=True)
    state = CanonicalMPS(state, normalize=True)
    E = H.expectation(state, state).real
    trajectory: list[float] = [E]
    Δβ_list: list[float] = [Δβ]
    β_list: list[float] = [0.0]
    results = EvolutionResults(
        state=state, energy=E, trajectory=trajectory, Δβ=Δβ_list, β=β_list
    )
    i = 0
    while i < maxiter:
        H_state = H.apply(state)
        k1 = -1 * H_state
        variance = k1.norm_squared()
        residual = sqrt(abs(variance - E * E))
        if residual < abs(tol_residual * E):
            break
        state2 = simplify(state + 0.25 * Δβ * k1, strategy=strategy)
        k2 = -1 * H.apply(state2)
        state3 = simplify(
            MPSSum([1, 3 * Δβ / 32, 9 * Δβ / 32], [state, k1, k2]), strategy=strategy
        )
        k3 = -1 * H.apply(state3)
        state4 = simplify(
            MPSSum(
                [1, (1932 / 2197) * Δβ, -(7200 / 2197) * Δβ, (7296 / 2197) * Δβ],
                [state, k1, k2, k3],
            ),
            strategy=strategy,
        )
        k4 = -1 * H.apply(state4)
        state5 = simplify(
            MPSSum(
                [1, (439 / 216) * Δβ, -8 * Δβ, (3680 / 513) * Δβ, -(845 / 4104) * Δβ],
                [state, k1, k2, k3, k4],
            ),
            strategy=strategy,
        )
        k5 = -1 * H.apply(state5)
        state6 = simplify(
            MPSSum(
                [
                    1,
                    -(8 / 27) * Δβ,
                    2 * Δβ,
                    -(3544 / 2565) * Δβ,
                    (1859 / 4104) * Δβ,
                    -(11 / 40) * Δβ,
                ],
                [state, k1, k2, k3, k4, k5],
            ),
            strategy=strategy,
        )
        k6 = -1 * H.apply(state6)
        state_ord5 = simplify(
            MPSSum(
                [
                    1,
                    (16 / 135) * Δβ,
                    (6656 / 12825) * Δβ,
                    (28561 / 56430) * Δβ,
                    -(9 / 50) * Δβ,
                    (2 / 55) * Δβ,
                ],
                [state, k1, k3, k4, k5, k6],
            ),
            strategy=normalization_strategy,
        )
        state_ord4 = simplify(
            MPSSum(
                [
                    1,
                    Δβ * (25 / 216),
                    Δβ * (1408 / 2565),
                    Δβ * (2197 / 4104),
                    Δβ * (-1 / 5),
                ],
                [state, k1, k3, k4, k5],
            ),
            strategy=normalization_strategy,
        )
        norm_ord5_sqr = state_ord5.norm_squared()
        norm_ord4_sqr = state_ord4.norm_squared()
        δ = sqrt(
            abs(norm_ord5_sqr + norm_ord4_sqr - 2 * scprod(state_ord5, state_ord4).real)
        ).real
        i += 1
        if δ > tol_rk:
            Δβ = 0.9 * Δβ * (tol_rk / δ) ** (1 / 5)
            continue
        E = H.expectation(state, state).real
        state = state_ord5
        trajectory.append(E)
        if callback is not None:
            callback(state, results)
        if E < results.energy:
            results.energy, results.state = E, state
            Δβ_list.append(Δβ)  # type: ignore
        β_list.append(β_list[-1] + Δβ)
        if δ > 0:
            Δβ = 0.9 * Δβ * (tol_rk / δ) ** (1 / 5)
    return results
