from __future__ import annotations
from typing import Callable, Any
import dataclasses
import numpy as np
from ..tools import make_logger
from ..state import MPS, CanonicalMPS, Strategy, random_mps, simplify
from ..operators import MPO, MPOList, MPOSum
from .descent import DESCENT_STRATEGY, OptimizeResults
from ..solve import cgs_solve


@dataclasses.dataclass
class PowerMethodOptimizeResults(OptimizeResults):
    steps: list[int] = dataclasses.field(default_factory=list)


def power_method(
    H: MPO | MPOList | MPOSum,
    inverse: bool = False,
    shift: float = 0.0,
    guess: MPS | None = None,
    maxiter: int = 1000,
    maxiter_cgs: int = 50,
    tol: float = 1e-13,
    tol_variance: float = 1e-14,
    tol_cgs: float = 1e-8,
    tol_up: float | None = None,
    upward_moves: int = 5,
    strategy: Strategy = DESCENT_STRATEGY,
    callback: Callable[[MPS, OptimizeResults], Any] | None = None,
) -> PowerMethodOptimizeResults:
    """Ground state search of Hamiltonian `H` by power method.

    Parameters
    ----------
    H : MPO | MPOList | MPOSum
        Hamiltonian in MPO form.
    guess : MPS | None
        Initial guess of the ground state. If None, defaults to a random
        MPS deduced from the operator's dimensions.
    maxiter : int
        Maximum number of iterations (defaults to 1000).
    maxiter_cgs : int
        Maximum number of iterations of CGS (defaults to 50).
    tol : float
        Energy variation that indicates termination (defaults to 1e-13).
    tol_up : float, default = `tol`
        If energy fluctuates up below this tolerance, continue the optimization.
    tol_variance : float
        Energy variance target (defaults to 1e-14).
    strategy : Strategy | None
        Linear combination of MPS truncation strategy. Defaults to
        DESCENT_STRATEGY.
    callback : Callable[[MPS, OptimizeResults], Any] | None
        A callable called after each iteration (defaults to None).

    Returns
    -------
    PowerMethodOptimizeResults
        Results from the optimization. See :class:`OptimizeResults`.
    """
    if tol_up is None:
        tol_up = tol
    if tol_cgs is None:
        tol_cgs = tol_variance
    if abs(shift):
        identity = MPO([np.eye(d).reshape(1, d, d, 1) for d in H.dimensions()])
        H = (H + shift * identity).join()
    state = CanonicalMPS(
        random_mps(H.dimensions(), D=2) if guess is None else guess,
        strategy=strategy,
    )
    results = PowerMethodOptimizeResults(
        state=state,
        energy=np.inf,
        converged=False,
        trajectory=[],
        variances=[],
        message=f"Maximum number of iterations {maxiter} reached",
    )
    # This extra field is needed because CGS consumes iterations
    # in itself.
    results.steps = []
    last_energy = np.inf
    logger = make_logger()
    logger(f"power_method() invoked with {maxiter} iterations")
    total_steps = 0

    def cgs_callback(state: MPS, residual: float) -> None:
        nonlocal total_steps
        total_steps += 1

    for step in range(maxiter):
        state.normalize_inplace()
        energy = H.expectation(state).real
        if energy < results.energy:
            results.energy, results.state = energy, state
        H_v = H @ state
        variance = abs(H_v.norm_squared() - energy * energy)
        results.trajectory.append(energy)
        results.variances.append(variance)
        results.steps.append(total_steps)
        logger(f"step = {step:5d}, energy = {energy}, variance = {variance}")
        if callback is not None:
            callback(state, results)
        energy_change = energy - last_energy
        if energy_change > tol_up:
            if upward_moves <= 0:
                results.message = (
                    f"Energy fluctuates upwards above tolerance {tol_up:5g}"
                )
                results.converged = True
                break
            logger("Upwards energy fluctuation ignored {energy_change:5g}")
            upward_moves -= 1
        if -abs(tol) < energy_change < 0:
            results.message = f"Energy converged within tolerance {tol:5g}"
            results.converged = True
            break
        last_energy = energy
        if variance < tol_variance:
            results.message = (
                f"Stationary state reached within tolerance {tol_variance:5g}"
            )
            results.converged = True
            break
        if total_steps > maxiter:
            break
        if inverse:
            state, res = cgs_solve(
                H,
                state,
                guess=(1 / energy) * state,
                maxiter=maxiter_cgs,
                tolerance=tol_cgs,
                strategy=strategy,
            )
            logger(f"CGS error = {res}")
        else:
            state = simplify(H_v, strategy=strategy)
            total_steps += 1
    logger(f"power_method() finished with results\n{results}")
    logger.close()
    return results
