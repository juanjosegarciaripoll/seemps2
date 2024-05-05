from __future__ import annotations
from typing import Callable, Union, Optional, Any
import dataclasses
import numpy as np
from ..tools import make_logger
from ..state import MPS, CanonicalMPS, Strategy, random_mps
from ..truncate import simplify
from ..mpo import MPO, MPOList, MPOSum
from .descent import DESCENT_STRATEGY, OptimizeResults
from ..cgs import cgs


@dataclasses.dataclass
class PowerMethodOptimizeResults(OptimizeResults):
    steps: list[int] = dataclasses.field(default_factory=list)


def power_method(
    H: Union[MPO, MPOList, MPOSum],
    inverse: bool = False,
    shift: float = 0.0,
    guess: Optional[MPS] = None,
    maxiter: int = 1000,
    maxiter_cgs: int = 50,
    tol: float = 1e-13,
    tol_variance: float = 1e-14,
    tol_cgs: Optional[float] = None,
    tol_up: Optional[float] = None,
    upward_moves: int = 5,
    strategy: Strategy = DESCENT_STRATEGY,
    callback: Optional[Callable[[MPS, OptimizeResults], Any]] = None,
) -> PowerMethodOptimizeResults:
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
    maxiter_cgs : int
        Maximum number of iterations of CGS (defaults to 50).
    tol : float
        Energy variation that indicates termination (defaults to 1e-13).
    tol_up : float, default = `tol`
        If energy fluctuates up below this tolerance, continue the optimization.
    tol_variance : float
        Energy variance target (defaults to 1e-14).
    strategy : Optional[Strategy]
        Linear combination of MPS truncation strategy. Defaults to
        DESCENT_STRATEGY.
    callback : Optional[Callable[[MPS, OptimizeResults], Any]]
        A callable called after each iteration (defaults to None).

    Results
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
        energy=np.Inf,
        converged=False,
        trajectory=[],
        variances=[],
        message=f"Maximum number of iterations {maxiter} reached",
    )
    # This extra field is needed because CGS consumes iterations
    # in itself.
    results.steps = []
    last_energy = np.Inf
    logger = make_logger()
    logger(f"power_method() invoked with {maxiter} iterations")
    total_steps = 0

    def cgs_callback(state, residual):
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
            print("Upwards energy fluctuation ignored {energy_change:5g}")
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
            state, residual = cgs(
                H,
                state,
                guess=(1 / energy) * state,
                maxiter=maxiter_cgs,
                tolerance=tol_cgs,
                strategy=strategy,
                callback=cgs_callback,
            )
        else:
            state = simplify(H_v, strategy=strategy)
            total_steps += 1
    logger(f"power_method() finished with results\n{results}")
    logger.close()
    return results
