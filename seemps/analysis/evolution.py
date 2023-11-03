from dataclasses import dataclass
from typing import Union

import numpy as np
from ..expectation import scprod
from ..state import MPS, CanonicalMPS
from ..typing import *
from ..optimization.descent import DESCENT_STRATEGY
from ..truncate.combine import combine


@dataclass
class EvolutionResults:
    """Results from ground state search using imaginary time evolution.

    Parameters
    ----------
    state : Union[MPS, np.ndarray]
        The estimate for the ground state.
    energy : float
        Estimate for the ground state energy.
    converged : bool
        True if the algorithm has found an approximate minimum, according
        to the given tolerance.
    message : str
        Message explaining why the algorithm stoped, both when it converged,
        and when it did not.
    trajectory : Optional[Vector]
        Vector of computed energies in the evolution trajectory.
    Δβ : float or List
        Steps size or steps sizes for each iteration.
    β : np.ndarray
        Imaginary time evolution path.
    """

    state: Union[MPS, np.ndarray]
    energy: float
    converged: bool
    message: str
    trajectory: Optional[VectorLike] = None
    Δβ: Union[float, VectorLike] = None
    β: Optional[VectorLike] = None

def euler(H, state, Δβ=0.01, maxiter=1000, tol: float = 1e-13, 
          strategy=DESCENT_STRATEGY, callback=None):
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
    tol : float
        Energy variation that indicates termination (defaults to 1e-13).
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
    energies = []
    last_E = np.Inf
    best_energy = np.Inf
    best_vector = state
    converged = False
    message = f"Maximum number of iterations {maxiter} reached"
    state = CanonicalMPS(state, normalize=True)
    for i in range(maxiter):
        H_state = H.apply(state)
        E = H.expectation(state).real
        energies.append(E)
        if E < best_energy:
            best_energy, best_vector = E, state
        if np.abs(E - last_E) < tol:
            message = f"Energy converged within tolerance {tol}"
            converged = True
            break
        last_E = E
        state = (state - Δβ * H_state)
        state = combine(state.weights, state.states, strategy=strategy)
        if callback is not None:
            callback(state)
    if not converged:
        H_state = H.apply(state)
        E = H.expectation(state).real
        energies.append(E)
        if E < best_energy:
            best_energy, best_vector = E, state
    β = Δβ * np.arange(maxiter + 1)
    return EvolutionResults(
            state=best_vector,
            energy=best_energy,
            converged=converged,
            message=message,
            trajectory=energies,
            Δβ=Δβ,
            β=β
    )

def improved_euler(H, state, Δβ=0.01, maxiter=1000, tol: float = 1e-13, 
                   strategy=DESCENT_STRATEGY, callback=None):
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
    tol : float
        Energy variation that indicates termination (defaults to 1e-13).
    strategy : Optional[Strategy]
        Truncation strategy when applying MPO. Defaults to DESCENT_STRATEGY.
    callback : Optional[callable]
        A callable called after each iteration (defaults to None).

    Results
    -------
    EvolutionResults
        Results from the evolution. See :class:`EvolutionResults`.
    """
    energies = []
    last_E = np.Inf
    best_energy = np.Inf
    best_vector = state
    converged = False
    message = f"Maximum number of iterations {maxiter} reached"
    state = CanonicalMPS(state, normalize=True)
    for i in range(maxiter):
        H_state = H.apply(state)
        E = H.expectation(state).real
        energies.append(E)
        if E < best_energy:
            best_energy, best_vector = E, state
        if np.abs(E - last_E) < tol:
            message = f"Energy converged within tolerance {tol}"
            converged = True
            break
        last_E = E
        k = 2 * state - Δβ * H_state
        Hk = H.apply(k)
        state = (state - 0.5 * Δβ * Hk)
        state = combine(state.weights, state.states, strategy=strategy)
        if callback is not None:
            callback(state)
    if not converged:
        H_state = H.apply(state)
        E = H.expectation(state).real
        energies.append(E)
        if E < best_energy:
            best_energy, best_vector = E, state
    β = Δβ * np.arange(maxiter + 1)
    return EvolutionResults(
            state=best_vector,
            energy=best_energy,
            converged=converged,
            message=message,
            trajectory=energies,
            Δβ=Δβ,
            β=β
    )

def runge_kutta(H, state, Δβ=0.01, maxiter=1000, tol: float = 1e-13, 
                strategy=DESCENT_STRATEGY, callback=None):
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
    tol : float
        Energy variation that indicates termination (defaults to 1e-13).
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
    energies = []
    last_E = np.Inf
    best_energy = np.Inf
    best_vector = state
    converged = False
    message = f"Maximum number of iterations {maxiter} reached"
    state = CanonicalMPS(state, normalize=True)
    for i in range(maxiter):
        H_state = H.apply(state)
        E = H.expectation(state).real
        energies.append(E)
        if E < best_energy:
            best_energy, best_vector = E, state
        if np.abs(E - last_E) < tol:
            message = f"Energy converged within tolerance {tol}"
            converged = True
            break
        last_E = E
        state2 = state - 0.5 * Δβ * H_state
        H_state2 = H.apply(state2)
        state3 = state - 0.5 * Δβ * H_state2
        H_state3 = H.apply(state3)
        state4 = state - Δβ * H_state3
        H_state4 = H.apply(state4)
        state = (state - Δβ / 6 * (H_state + 2 * H_state2 + 2 * H_state3 + H_state4))
        state = combine(state.weights, state.states, strategy=strategy)
        if callback is not None:
            callback(state)
    if not converged:
        H_state = H.apply(state)
        E = H.expectation(state).real
        energies.append(E)
        if E < best_energy:
            best_energy, best_vector = E, state
    β = Δβ * np.arange(maxiter + 1)
    return EvolutionResults(
            state=best_vector,
            energy=best_energy,
            converged=converged,
            message=message,
            trajectory=energies,
            Δβ=Δβ,
            β=β
    )

def runge_kutta_fehlberg(H, state, Δβ=0.01, maxiter=1000, tol: float = 1e-13, 
                         tol_rk: float = 1e-8, strategy=DESCENT_STRATEGY, callback=None):
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
    tol : float
        Energy variation that indicates termination (defaults to 1e-13).
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
    energies = []
    Δβs = []
    last_E = np.Inf
    best_energy = np.Inf
    best_vector = state
    converged = False
    message = f"Maximum number of iterations {maxiter} reached"
    i = 0
    state = CanonicalMPS(state, normalize=True)
    while i < maxiter:
        H_state = H.apply(state)
        E = H.expectation(state).real
        k1 = -1 * H_state
        state2 = state + 0.25 * Δβ * k1
        k2 = -1 * H.apply(state2)
        state3 = state + (3 / 32) * Δβ * k1 + (9 / 32) * Δβ * k2
        k3 = -1 * H.apply(state3)
        state4 = (
            state
            + (1932 / 2197) * Δβ * k1
            - (7200 / 2197) * Δβ * k2
            + (7296 / 2197) * Δβ * k3
        )
        k4 = -1 * H.apply(state4)
        state5 = (
            state
            + (439 / 216) * Δβ * k1
            - 8 * Δβ * k2
            + (3680 / 513) * Δβ * k3
            - (845 / 4104) * Δβ * k4
        )
        k5 = -1 * H.apply(state5)
        state6 = (
            state
            - (8 / 27) * Δβ * k1
            + 2 * Δβ * k2
            - (3544 / 2565) * Δβ * k3
            + (1859 / 4104) * Δβ * k4
            - (11 / 40) * Δβ * k5
        )
        k6 = -1 * H.apply(state6)
        state_ord5 = (state + Δβ * (
            (16 / 135) * k1
            + (6656 / 12825) * k3
            + (28561 / 56430) * k4
            - (9 / 50) * k5
            + (2 / 55) * k6
        ))
        state_ord5 = combine(state_ord5.weights, state_ord5.states, strategy=strategy)
        state_ord4 = (state + Δβ * (
            (25 / 216) * k1 + (1408 / 2565) * k3 + (2197 / 4104) * k4 - (1 / 5) * k5
        ))
        state_ord4 = combine(state_ord4.weights, state_ord4.states, strategy=strategy)
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
            if np.abs(E - last_E) < tol:
                message = f"Energy converged within tolerance {tol}"
                converged = True
                break
            state = state_ord5
            if callback is not None:
                callback(state)
            last_E = E
            Δβs.append(Δβ)
        else:
            i += 1
            energies.append(E)
            if E < best_energy:
                best_energy, best_vector = E, state
            if np.abs(E - last_E) < tol:
                message = f"Energy converged within tolerance {tol}"
                converged = True
                break
            state = state_ord5
            if callback is not None:
                callback(state)
            last_E = E
            Δβs.append(Δβ)
            Δβ = 0.9 * Δβ * (tol_rk / δ) ** (1 / 5)
    if not converged:
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
            converged=converged,
            message=message,
            trajectory=energies,
            Δβ=Δβs,
            β=β
    )