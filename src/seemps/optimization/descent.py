from dataclasses import dataclass
from typing import Callable, Union

import numpy as np

from ..expectation import scprod
from ..mpo import MPO, MPOList, MPOSum
from ..state import DEFAULT_STRATEGY, MPS, CanonicalMPS, Simplification, Strategy
from ..tools import DEBUG, log
from ..truncate.simplify import simplify
from ..typing import *

DESCENT_STRATEGY = DEFAULT_STRATEGY.replace(simplify=Simplification.VARIATIONAL)


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
    H: Union[MPO, MPOList, MPOSum],
    state: MPS,
    maxiter=1000,
    tol: float = 1e-13,
    tol_up: float = 1e-13,
    k_mean=10,
    tol_variance: float = 1e-14,
    strategy: Strategy = DESCENT_STRATEGY,
    callback: Optional[Callable] = None,
) -> OptimizeResults:
    """Ground state search of Hamiltonian `H` by gradient descent.

    Parameters
    ----------
    H : Union[MPO, MPOList, MPOSum]
        Hamiltonian in MPO form.
    state : MPS
        Initial guess of the ground state.
    maxiter : int
        Maximum number of iterations (defaults to 1000).
    tol : float
        Energy variation with respect to the k_mean moving average that
        indicates termination (defaults to 1e-13).
    k_mean: int
        Number of elements for the moving average.
    tol_variance : float
        Energy variance target (defaults to 1e-14).
    strategy : Optional[Strategy]
        Linear combination of MPS truncation strategy. Defaults to
        DESCENT_STRATEGY.
    callback : Optional[callable]
        A callable called after each iteration (defaults to None).

    Results
    -------
    OptimizeResults
        Results from the optimization. See :class:`OptimizeResults`.
    """

    def energy_and_variance(state: MPS) -> tuple[MPS, float, float, float]:
        true_E = H.expectation(state).real
        H_state: MPS = H.apply(state)  # type: ignore
        avg_H2 = scprod(H_state, H_state).real
        variance = avg_H2 - scprod(state, H_state).real ** 2
        return H_state, true_E, variance, avg_H2

    normalization_strategy = strategy.replace(normalize=True)
    energies: list[float] = []
    variances: list[float] = []
    last_E_mean: float = np.Inf
    best_energy: float = np.Inf
    best_vector: MPS = state
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
    state = CanonicalMPS(state, normalize=True)
    for step in range(maxiter):
        H_state, E, variance, avg_H2 = energy_and_variance(state)
        if callback is not None:
            callback(state)
        if DEBUG:
            log(f"step = {step:5d}, energy = {E}, variance = {variance}")
        energies.append(E)
        variances.append(variance)
        if E < best_energy:
            best_energy, best_vector, _ = E, state, variance
        E_mean: float = np.mean(energies[(-max(-k_mean - 1, len(energies))) : -1])  # type: ignore
        if (E_mean - last_E_mean > 0 and E_mean - last_E_mean >= abs(tol_up)) or (E_mean - last_E_mean <0 and  E_mean - last_E_mean>= -abs(tol)):
            message = f"Energy converged within tolerance {tol}"
            converged = True
            break
        if variance < tol_variance:
            message = f"Stationary state reached within tolerance {tol_variance}"
            converged = True
            break
        avg_H3 = H.expectation(H_state).real
        A = np.array([[E, avg_H2], [avg_H2, avg_H3]])
        B = np.array([[1, E], [E, avg_H2]])
        w, v = scipy.linalg.eig(A, B)
        v = v[:, np.argmin(w)]
        v /= np.linalg.norm(v)
        state = simplify(v[0] * state + v[1] * H_state, strategy=normalization_strategy)
        last_E_mean = E_mean
        # TODO: Implement stop criteria based on gradient size Δβ
        # It must take into account the norm of the displacement, H_state
        # which was already calculated
    if not converged:
        H_state, E, variance, _ = energy_and_variance(state)
        if callback is not None:
            callback(state)
        if E < best_energy:
            best_energy, best_vector, _ = E, state, variance
        energies.append(E)
        variances.append(variance)
    return OptimizeResults(
        state=best_vector,
        energy=best_energy,
        converged=converged,
        message=message,
        trajectory=energies,
        variances=variances,
    )
