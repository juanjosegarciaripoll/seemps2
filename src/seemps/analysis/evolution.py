from dataclasses import dataclass
from typing import Union

import numpy as np

from ..expectation import scprod
from ..optimization.descent import DESCENT_STRATEGY
from ..state import MPS, CanonicalMPS
from ..truncate.simplify import simplify
from ..typing import *


@dataclass
class EvolutionResults:
    """Results from ground state search using imaginary time evolution.

    Parameters
    ----------
    state : Union[MPS, np.ndarray]
        The estimate for the ground state.
    energy : float
        Estimate for the ground state energy.
    trajectory : Optional[Vector]
        Vector of computed energies in the evolution trajectory.
    Δβ : float or List
        Steps size or steps sizes for each iteration.
    β : np.ndarray
        Imaginary time evolution path.
    """

    state: Union[MPS, np.ndarray]
    energy: float
    trajectory: Optional[VectorLike] = None
    Δβ: Union[float, VectorLike, None] = None
    β: Optional[VectorLike] = None


def euler(
    H,
    state,
    Δβ=0.01,
    maxiter=1000,
    strategy=DESCENT_STRATEGY,
    callback=None,
):
    """Euler method for arrays.

    Parameters
    ----------
    H : MPO
        Hamiltonian in MPO form.
    state : MPS
        Initial guess of the ground state.
    Δβ : float
        Step size (defaults to 0.01).
    maxiter : int
        Maximum number of iterations (defaults to 1000)
    strategy : Optional[Strategy]
        Truncation strategy when applying MPO. Defaults to DESCENT_STRATEGY, thereby
        using whatever strategy the MPO has defined.
    callback : Optional[callable]
        A callable called after each iteration (defaults to None).

    Results
    -------
    EvolutionResults
        Results from the evolution. See :class:`EvolutionResults`.
    """
    normalization_strategy = strategy.replace(normalize=True)
    energies = []
    best_energy = np.Inf
    best_vector = state
    state = CanonicalMPS(state, normalize=True)
    for i in range(maxiter):
        H_state = H.apply(state)
        E = H.expectation(state).real
        if callback is not None:
            callback(state)
        energies.append(E)
        if E < best_energy:
            best_energy, best_vector = E, state
        state = simplify(state - Δβ * H_state, strategy=normalization_strategy)
    H_state = H.apply(state)
    E = H.expectation(state).real
    energies.append(E)
    if E < best_energy:
        best_energy, best_vector = E, state
    β = Δβ * np.arange(maxiter + 1)
    return EvolutionResults(
        state=best_vector,
        energy=best_energy,
        trajectory=energies,
        Δβ=Δβ,
        β=β,
    )


def improved_euler(
    H,
    state,
    Δβ=0.01,
    maxiter=1000,
    strategy=DESCENT_STRATEGY,
    callback=None,
):
    """Improved Euler method for arrays.

    Parameters
    ----------
    H : MPO
        Hamiltonian in MPO form.
    state : MPS
        Initial guess of the ground state.
    Δβ : float
        Step size (defaults to 0.01).
    maxiter : int
        Maximum number of iterations (defaults to 1000)
    strategy : Optional[Strategy]
        Truncation strategy when applying MPO. Defaults to DESCENT_STRATEGY.
    callback : Optional[callable]
        A callable called after each iteration (defaults to None).

    Results
    -------
    EvolutionResults
        Results from the evolution. See :class:`EvolutionResults`.
    """
    normalization_strategy = strategy.replace(normalize=True)
    energies = []
    best_energy = np.Inf
    best_vector = state
    state = CanonicalMPS(state, normalize=True)
    for i in range(maxiter):
        H_state = H.apply(state)
        E = H.expectation(state).real
        if callback is not None:
<<<<<<< HEAD
            callback(state)
=======
            callback(state, E)
>>>>>>> efa1ec87dedd2493560275876751c3699594fafd
        energies.append(E)
        if E < best_energy:
            best_energy, best_vector = E, state
        k = simplify(2 * state - Δβ * H_state, strategy=strategy)
        Hk = H.apply(k)
        state = simplify(state - 0.5 * Δβ * Hk, strategy=normalization_strategy)
    H_state = H.apply(state)
    E = H.expectation(state).real
    energies.append(E)
    if E < best_energy:
        best_energy, best_vector = E, state
    β = Δβ * np.arange(maxiter + 1)
    return EvolutionResults(
        state=best_vector,
        energy=best_energy,
        trajectory=energies,
        Δβ=Δβ,
        β=β,
    )


def runge_kutta(
    H,
    state,
    Δβ=0.01,
    maxiter=1000,
    strategy=DESCENT_STRATEGY,
    callback=None,
):
    """Runge-Kutta method for arrays.

    Parameters
    ----------
    H : MPO
        Hamiltonian in MPO form.
    state : MPS
        Initial guess of the ground state.
    Δβ : float
        Step size (defaults to 0.01).
    maxiter : int
        Maximum number of iterations (defaults to 1000)
    strategy : Optional[Strategy]
        Truncation strategy when applying MPO. Defaults to DESCENT_STRATEGY, thereby
        using whatever strategy the MPO has defined.
    callback : Optional[callable]
        A callable called after each iteration (defaults to None).

    Results
    -------
    EvolutionResults
        Results from the evolution. See :class:`EvolutionResults`.
    """
    normalization_strategy = strategy.replace(normalize=True)
    energies = []
    best_energy = np.Inf
    best_vector = state
    state = CanonicalMPS(state, normalize=True)
    for i in range(maxiter):
        H_state = H.apply(state)
        E = H.expectation(state).real
        if callback is not None:
<<<<<<< HEAD
            callback(state)
=======
            callback(state, E)
>>>>>>> efa1ec87dedd2493560275876751c3699594fafd
        energies.append(E)
        if E < best_energy:
            best_energy, best_vector = E, state
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
    H_state = H.apply(state)
    E = H.expectation(state).real
    energies.append(E)
    if E < best_energy:
        best_energy, best_vector = E, state
    β = Δβ * np.arange(maxiter + 1)
    return EvolutionResults(
        state=best_vector,
        energy=best_energy,
        trajectory=energies,
        Δβ=Δβ,
        β=β,
    )


def runge_kutta_fehlberg(
    H,
    state,
    Δβ=0.01,
    maxiter=1000,
    tol_rk: float = 1e-8,
    strategy=DESCENT_STRATEGY,
    callback=None,
):
    """Runge-Kutta method for arrays.

    Parameters
    ----------
    H : MPO
        Hamiltonian in MPO form.
    state : MPS
        Initial guess of the ground state.
    Δβ : float
        Step size (defaults to 0.01).
    maxiter : int
        Maximum number of iterations (defaults to 1000)
    tol_rk : float
        Energy variation that indicates termination (defaults to 1e-8).
    strategy : Optional[Strategy]
        Truncation strategy when applying MPO. Defaults to DESCENT_STRATEGY, thereby
        using whatever strategy the MPO has defined.
    callback : Optional[callable]
        A callable called after each iteration (defaults to None).

    Results
    -------
    EvolutionResults
        Results from the evolution. See :class:`EvolutionResults`.
    """
    normalization_strategy = strategy.replace(normalize=True)
    energies = []
    Δβs = []
    best_energy = np.Inf
    best_vector = state
    i = 0
    state = CanonicalMPS(state, normalize=True)
    while i < maxiter:
        H_state = H.apply(state)
        E = H.expectation(state).real
        if callback is not None:
<<<<<<< HEAD
            callback(state)
=======
            callback(state, E)
>>>>>>> efa1ec87dedd2493560275876751c3699594fafd
        k1 = -1 * H_state
        state2 = simplify(state + 0.25 * Δβ * k1, strategy=strategy)
        k2 = -1 * H.apply(state2)
        state3 = simplify(
            state + (3 / 32) * Δβ * k1 + (9 / 32) * Δβ * k2, strategy=strategy
        )
        k3 = -1 * H.apply(state3)
        state4 = simplify(
            state
            + (1932 / 2197) * Δβ * k1
            - (7200 / 2197) * Δβ * k2
            + (7296 / 2197) * Δβ * k3,
            strategy=strategy,
        )
        k4 = -1 * H.apply(state4)
        state5 = simplify(
            state
            + (439 / 216) * Δβ * k1
            - 8 * Δβ * k2
            + (3680 / 513) * Δβ * k3
            - (845 / 4104) * Δβ * k4,
            strategy=strategy,
        )
        k5 = -1 * H.apply(state5)
        state6 = simplify(
            state
            - (8 / 27) * Δβ * k1
            + 2 * Δβ * k2
            - (3544 / 2565) * Δβ * k3
            + (1859 / 4104) * Δβ * k4
            - (11 / 40) * Δβ * k5,
            strategy=strategy,
        )
        k6 = -1 * H.apply(state6)
        state_ord5 = simplify(
            state
            + Δβ
            * (
                (16 / 135) * k1
                + (6656 / 12825) * k3
                + (28561 / 56430) * k4
                - (9 / 50) * k5
                + (2 / 55) * k6
            ),
            strategy=normalization_strategy,
        )
        state_ord4 = simplify(
            state
            + Δβ
            * (
                (25 / 216) * k1 + (1408 / 2565) * k3 + (2197 / 4104) * k4 - (1 / 5) * k5
            ),
            strategy=normalization_strategy,
        )
        δ = np.sqrt(
            abs(
                scprod(state_ord5, state_ord5)
                + scprod(state_ord4, state_ord4)
                - 2 * scprod(state_ord5, state_ord4).real
            )
        ).real
        if δ > tol_rk:
            Δβ = 0.9 * Δβ * (tol_rk / δ) ** (1 / 5)
        elif δ == 0:
            i += 1
            energies.append(E)
            if E < best_energy:
                best_energy, best_vector = E, state
            state = state_ord5
            Δβs.append(Δβ)
        else:
            if callback is not None:
                callback(None)
            i += 1
            energies.append(E)
            if E < best_energy:
                best_energy, best_vector = E, state
            state = state_ord5
            Δβs.append(Δβ)
            Δβ = 0.9 * Δβ * (tol_rk / δ) ** (1 / 5)
    H_state = H.apply(state)
    E = H.expectation(state).real
    energies.append(E)
    if E < best_energy:
        best_energy, best_vector = E, state
    β = [0]
    β_i = 0
    for Δβ_i in Δβs:
        β_i += Δβ_i
        β.append(β_i)
    return EvolutionResults(
        state=best_vector,
        energy=best_energy,
        trajectory=energies,
        Δβ=Δβs,
        β=β,
    )
