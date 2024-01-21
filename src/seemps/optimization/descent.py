from __future__ import annotations
import dataclasses
from typing import Callable, Union, Any

import numpy as np

from ..expectation import scprod
from ..mpo import MPO, MPOList, MPOSum
from ..state import (
    DEFAULT_STRATEGY,
    MPS,
    MPSSum,
    CanonicalMPS,
    Simplification,
    Strategy,
)
from ..tools import DEBUG, log
from ..truncate.simplify import simplify
from ..typing import *

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
    trajectory : Optional[Vector]
        Vector of computed energies in the optimization trajectory.
    variances : Optional[Vector]
        Vector of computed energy variance in the optimization trajectory.
    """

    state: MPS
    energy: float
    converged: bool
    message: str
    trajectory: list[float] = dataclasses.field(default_factory=list)
    variances: list[float] = dataclasses.field(default_factory=list)


def gradient_descent(
    H: Union[MPO, MPOList, MPOSum],
    state: MPS,
    maxiter=1000,
    tol: float = 1e-13,
    tol_variance: float = 1e-14,
    strategy: Strategy = DESCENT_STRATEGY,
    callback: Optional[Callable[[MPS, float, OptimizeResults], Any]] = None,
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
        Minimal energy variation that indicates termination (defaults to 1e-13).
    tol_variance : float
        Energy variance target (defaults to 1e-14).
    strategy : Optional[Strategy]
        Linear combination of MPS truncation strategy. Defaults to
        DESCENT_STRATEGY.
    callback : Optional[Callable[[MPS, float, OptimizeResult], Any]]
        A callable called after each iteration with the current state,
        an estimate of the energy, and the accumulated results object.
        Defaults to None.

    Results
    -------
    OptimizeResults
        Results from the optimization. See :class:`OptimizeResults`.
    """

    normalization_strategy = strategy.replace(normalize=True)
    last_E: float = np.Inf
    """
    The algorithm aims to find the optimal linear combination
        ψ' = a * ψ + b * H * ψ
    that minimizes the energy <ψ'|H|ψ'>/<ψ'|ψ'>. This is equivalent to solving
    the generalized eigenvalue problem

        | <ψ|H|ψ>    <ψ|H*H|ψ>   | | a |     | <ψ|ψ>    <ψ|H|ψ>   |
        |                        | |   | = l |                    |
        | <ψ|H*H|ψ>  <ψ|H*H*H|ψ> | | b |     | <ψ|H|ψ>  <ψ|H*H|ψ> |

    """
    state = CanonicalMPS(state, normalize=True)
    results = OptimizeResults(
        energy=np.Inf,
        state=state,
        converged=False,
        message=f"Maximum number of iterations {maxiter} reached",
    )
    for step in range(maxiter + 1):
        if step:
            state = simplify(
                MPSSum(weights, [state, H_state]),  # type: ignore
                strategy=normalization_strategy,
            )
        E = H.expectation(state).real
        H_state: MPS = H.apply(state)  # type: ignore
        avg_H2 = scprod(H_state, H_state).real
        variance = avg_H2 - E * E
        if callback is not None:
            callback(state, E, results)
        if DEBUG:
            log(f"step = {step:5d}, energy = {E}, variance = {variance}")
        results.trajectory.append(E)
        results.variances.append(variance)
        if E < results.energy:
            results.energy, results.state = E, state
        if E - last_E >= -abs(tol):
            results.message = f"Energy converged within tolerance {tol}"
            results.converged = True
            break
        last_E = E
        if variance < tol_variance:
            results.message = (
                f"Stationary state reached within tolerance {tol_variance}"
            )
            results.converged = True
            break
        avg_H3 = H.expectation(H_state).real
        w, eigenvectors = scipy.linalg.eig(
            [[E, avg_H2], [avg_H2, avg_H3]], [[1, E], [E, avg_H2]]
        )
        weights = eigenvectors[:, np.argmin(w)]

    return results
