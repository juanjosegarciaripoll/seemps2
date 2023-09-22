import numpy as np
from ..state import CanonicalMPS, MPS, Strategy
from ..expectation import scprod
from ..mpo import MPO
from ..typing import *
from dataclasses import dataclass
from ..tools import log


@dataclass
class OptimizeResults:
    """Results from ground state search.

    Parameters
    ----------
    state : MPS
        The estimate for the ground state.
    energy : float
        Estimate for the ground state energy.
    converged : bool
        True if the algorithm has found an approximate minimum, according
        to the given tolerances.
    message : str
        Message explaining why the algorithm stoped, both when it converged,
        and when it did not.
    trajectory : Optional[Vector]
        Vector of computed energies in the optimization trajectory.
    variances : Optional[Vector]
        Vector of computed energy variance in the optimization trajectory.
    """

    state: MPS
    energy: float
    converged: bool
    message: str
    trajectory: Optional[VectorLike] = None
    variances: Optional[VectorLike] = None


def gradient_descent(
    H: MPO,
    state: MPS,
    maxiter=1000,
    tol: float = 1e-13,
    tol_variance: float = 1e-14,
    strategy: Optional[Strategy] = None,
) -> OptimizeResults:
    """Ground state search of Hamiltonian `H` by gradient descent.

    Parameters
    ----------
    ψ : MPS
        Initial guess of the ground state.
    H : MPO
        Hamiltonian in MPO form.
    tol_variance : float
        Energy variance target (defaults to 1e-14)
    tol : float
        Energy variation that indicates termination (defaults to 1e-13).
    strategy : Optional[Strategy]
        Truncation strategy when applying MPO. Defaults to `None`, thereby
        using whatever strategy the MPO has defined.

    Results
    -------
    OptimizeResults
        Results from the optimization. See :class:`OptimizeResults`.
    """

    def energy_and_variance(state: MPS) -> tuple[MPS, float, float, float]:
        true_E = H.expectation(state).real
        H_state = H.apply(state, strategy=strategy)
        avg_H2 = scprod(H_state, H_state).real
        variance = avg_H2 - scprod(state, H_state).real ** 2
        return H_state, true_E, variance, avg_H2

    energies = []
    variances = []
    last_E = np.Inf
    best_energy = np.Inf
    best_variance = np.Inf
    best_vector = state
    message = "Maximum number of iterations reached"

    """
    The algorithm aims to find the optimal linear combination
        ψ' = a * ψ + b * H * ψ
    that minimizes the energy <ψ'|H|ψ'>/<ψ'|ψ'>. This is equivalent to solving
    the generalized eigenvalue problem

        | <ψ|H|ψ>    <ψ|H*H|ψ>   | | a |     | <ψ|ψ>    <ψ|H|ψ>   |
        |                        | |   | = l |                    |
        | <ψ|H*H|ψ>  <ψ|H*H*H|ψ> | | b |     | <ψ|H|ψ>  <ψ|H*H|ψ> |
    
    """
    converged = False
    message = f"Maximum number of iterations {maxiter} reached"
    for step in range(maxiter):
        state = CanonicalMPS(state, normalize=True)
        H_state, E, variance, avg_H2 = energy_and_variance(state)
        log(f"step = {step:5d}, energy = {E}, variance = {variance}")
        energies.append(E)
        variances.append(variance)
        if E < best_energy:
            best_energy, best_vector, best_variance = E, state, variance
        if np.abs(E - last_E) < tol:
            message = f"Energy converged within tolerance {tol}"
            converged = True
            break
        if variance < tol_variance:
            message = f"Stationary state reached within tolerance {tol_variance}"
            converged = True
            break
        last_E = E
        avg_H3 = H.expectation(H_state).real
        avg_3 = (avg_H3 - 3 * E * avg_H2 + 2 * E**3).real
        Δβ = (avg_3 - np.sqrt(avg_3**2 + 4 * variance**3)) / (2 * variance**2)
        # TODO: Replace this formula with the formula that keeps the
        # normalization of the state (2nd. order gradient descent from the
        # manuscript)
        # TODO: Use directly `combine`
        state = (state + Δβ * (H_state - E * state)).toMPS()
        # TODO: Implement stop criteria based on gradient size Δβ
        # It must take into account the norm of the displacement, H_state
        # which was already calculated

    state = CanonicalMPS(state, normalize=True)
    H_state, E, variance, _ = energy_and_variance(state)
    if E > best_energy:
        E, state, variance = best_energy, best_vector, best_variance
    energies.append(E)
    variances.append(variance)
    return OptimizeResults(
        state=state,
        energy=E,
        converged=converged,
        message=message,
        trajectory=energies,
        variances=variances,
    )
