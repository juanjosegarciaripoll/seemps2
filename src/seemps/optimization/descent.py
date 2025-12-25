from __future__ import annotations
from typing import Callable, Any
import scipy.linalg
import dataclasses
import numpy as np
from ..tools import make_logger
from ..state import (
    DEFAULT_STRATEGY,
    MPS,
    MPSSum,
    Simplification,
    Strategy,
    scprod,
    to_mps,
    simplify_mps,
)
from ..operators import MPO, MPOList, MPOSum

DESCENT_STRATEGY = DEFAULT_STRATEGY.replace(simplify=Simplification.VARIATIONAL)


@dataclasses.dataclass
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
    trajectory : Vector | None
        Vector of computed energies in the optimization trajectory.
    variances : Vector | None
        Vector of computed energy variance in the optimization trajectory.
    """

    state: MPS
    energy: float
    converged: bool
    message: str
    trajectory: list[float] = dataclasses.field(default_factory=list)
    variances: list[float] = dataclasses.field(default_factory=list)


def gradient_descent(
    H: MPO | MPOList | MPOSum,
    guess: MPS | MPSSum,
    maxiter: int = 1000,
    tol: float = 1e-13,
    tol_up: float | None = None,
    tol_variance: float = 1e-14,
    strategy: Strategy = DESCENT_STRATEGY,
    callback: Callable[[MPS, OptimizeResults], Any] | None = None,
) -> OptimizeResults:
    """Ground state search of Hamiltonian `H` by gradient descent.

    Parameters
    ----------
    H : MPO | MPOList | MPOSum
        Hamiltonian in MPO form.
    state : MPS | MPSSum
        Initial guess of the ground state.
    maxiter : int
        Maximum number of iterations (defaults to 1000).
    tol : float
        Energy variation that indicates termination (defaults to 1e-13).
    tol_up : float, default = `tol`
        If energy fluctuates up below this tolerance, continue the optimization.
    tol_variance : float
        Energy variance target (defaults to 1e-14).
    strategy : Strategy, default = DESCENT_STRATEGY
        Linear combination of MPS truncation strategy.
    callback : Callable[[MPS, OptimizeResults], Any] | None
        A callable called after each iteration (defaults to None).

    Returns
    -------
    OptimizeResults
        Results from the optimization. See :class:`OptimizeResults`.
    """
    if maxiter < 1:
        raise Exception("maxiter cannot be zero or negative")
    if tol_up is None:
        tol_up = tol
    normalization_strategy = strategy.replace(normalize=True)
    state = simplify_mps(guess, strategy=strategy)
    results = OptimizeResults(
        state=state,
        energy=np.inf,
        converged=False,
        message=f"Exceeded maximum number of steps {maxiter}",
    )
    E = last_E = variance = avg_H2 = np.inf
    H_state: MPS
    with make_logger() as logger:
        logger(f"gradient_descent() invoked with {maxiter} iterations")
        for step in range(maxiter + 1):
            """
            The algorithm aims to find the optimal linear combination
                ψ' = a * ψ + b * H * ψ
            that minimizes the energy <ψ'|H|ψ'>/<ψ'|ψ'>. This is equivalent to solving
            the generalized eigenvalue problem

                | <ψ|H|ψ>    <ψ|H*H|ψ>   | | a |     | <ψ|ψ>    <ψ|H|ψ>   |
                |                        | |   | = l |                    |
                | <ψ|H*H|ψ>  <ψ|H*H*H|ψ> | | b |     | <ψ|H|ψ>  <ψ|H*H|ψ> |

            """
            E = H.expectation(state).real
            # TODO: We need a more powerful function that acts on MPO's
            # MPOList's and MPOSum's and returns an object that, when
            # applied onto an MPS always returns an MPS.
            H_state = to_mps(H.apply(state))
            avg_H2 = scprod(H_state, H_state).real
            variance = avg_H2 - scprod(state, H_state).real ** 2
            if callback is not None:
                callback(state, results)
            logger(f"step = {step:5d}, energy = {E}, variance = {variance}")
            results.trajectory.append(E)
            results.variances.append(variance)
            if E < results.energy:
                results.energy, results.state = E, state
            energy_change = E - last_E
            if energy_change > tol_up:
                results.message = (
                    f"Energy fluctuates upwards above tolerance {tol_up:5g}"
                )
                results.converged = True
                break
            if -abs(tol) < energy_change < 0:
                results.message = f"Energy converged within tolerance {tol:5g}"
                results.converged = True
                break
            last_E = E
            if variance < tol_variance:
                results.message = (
                    f"Stationary state reached within tolerance {tol_variance:5g}"
                )
                results.converged = True
                break
            if step <= maxiter:
                avg_H3 = H.expectation(H_state).real
                A = [[E, avg_H2], [avg_H2, avg_H3]]
                B = [[1, E], [E, avg_H2]]
                w, v = scipy.linalg.eig(A, B)  # type: ignore # (Pylance eror)
                v = v[:, np.argmin(w)]
                v /= np.linalg.norm(v)
                state = simplify_mps(
                    MPSSum(v, [state, H_state]), strategy=normalization_strategy
                )
        logger(
            f"Algorithm finished with:\nmessage={results.message}\nconverged = {results.converged}"
        )
    return results
