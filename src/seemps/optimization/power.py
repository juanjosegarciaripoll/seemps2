from dataclasses import dataclass
from typing import Callable, Union

import numpy as np

from ..expectation import scprod
from ..mpo import MPO, MPOList, MPOSum
from ..state import DEFAULT_STRATEGY, MPS, CanonicalMPS, Simplification, Strategy
from ..tools import DEBUG, log
from ..truncate import simplify
from .descent import DESCENT_STRATEGY, OptimizeResults
from ..cgs import cgs


def power_method(
    H: Union[MPO, MPOList, MPOSum],
    inverse: bool = False,
    guess: Optional[MPS] = None,
    maxiter: int = 1000,
    tol: float = 1e-13,
    tol_variance: float = 1e-14,
    strategy: Strategy = DESCENT_STRATEGY,
    callback: Optional[Callable] = None,
) -> OptimizeResults:
    """Ground state search of Hamiltonian `H` by power method.

    Parameters
    ----------
    H : Union[MPO, MPOList, MPOSum]
        Hamiltonian in MPO form.
    guess : Optional[MPS]
        Initial guess of the ground state. If None, defaults to a random
        MPS deduced from the operator's dimensions.
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
    state = CanonicalForm(
        random_mps(operator.dimensions(), D=2) if guess is None else guess,
        strategy=strategy,
    )
    results = OptimizeResults(
        state=state,
        energy=np.Inf,
        converged=False,
        message=f"Maximum number of iterations {maxiter} reached",
    )
    for step in range(maxiter):
        state.normalize_in_place()
        energy = H.expectation(state)
        if energy < results.energy:
            results.energy, results.state = energy, state
        H_v = H @ state
        variance = abs(scprod(H_v, H_v) - energy * energy)
        results.trajectory.append(energy)
        results.trajectory.append(variance)
        print(f"step = {step:5d}, energy = {E}, variance = {variance}")
        if DEBUG:
            log(f"step = {step:5d}, energy = {E}, variance = {variance}")
        if callback is not None:
            callback(state, results, energy)
        if inverse:
            state = cgs(H, state, guess=state, tolerance=tol, strategy=strategy)
        else:
            state = CanonicalForm(H_v, strategy=strategy)
        if energy - last_energy >= -abs(tol):
            results.message = f"Energy converged within tolerance {tol}"
            results.converged = True
            break
        if variance < tol_variance:
            results.message = (
                f"Stationary state reached within tolerance {tol_variance}"
            )
            results.converged = True
            break
    return results
