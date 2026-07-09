from __future__ import annotations
from typing import Callable
import numpy as np
import scipy.sparse.linalg
from ..tools import make_logger
from ..typing import Tensor4
from ..state import DEFAULT_STRATEGY, MPS, CanonicalMPS, Strategy, random_mps
from ..cython import _contract_last_and_first
from ..operators import MPO
from ..operators.quadratic import QuadraticForm
from ..hamiltonians import NNHamiltonian
from .descent import OptimizeResults


def _diagonalize_two_site(
    QF: QuadraticForm, i: int, tol_eigs: float
) -> tuple[float, Tensor4]:
    Op = QF.two_site_operator(i)
    v = _contract_last_and_first(QF.state[i], QF.state[i + 1])
    eval, evec = scipy.sparse.linalg.eigsh(
        Op, 1, which="SA", v0=v.reshape(-1), tol=tol_eigs
    )
    return eval[0], evec.reshape(v.shape)


def _sweep(
    QF: QuadraticForm,
    direction: int,
    tol_eigs: float,
    strategy: Strategy,
) -> float:
    """One full two-site sweep updating `QF` in place."""
    size = QF.state.size
    sites = range(size - 1) if direction > 0 else range(size - 2, -1, -1)
    E = np.inf
    for i in sites:
        E, AB = _diagonalize_two_site(QF, i, tol_eigs)
        if direction > 0:
            QF.update_2site_right(AB, i, strategy)
        else:
            QF.update_2site_left(AB, i, strategy)
    return E


def _convergence_reason(
    energy_change: float, energy_scale: float, tol: float, tol_up: float
) -> str | None:
    if energy_change > tol_up * energy_scale:
        return f"Energy fluctuation above tolerance {tol_up}"
    if -tol * energy_scale <= energy_change <= 0:
        return f"Energy decrease slower than tolerance {tol}"
    return None


def _energy_and_variance(H: MPO, state: MPS) -> tuple[float, float]:
    H_state = H.apply(state, simplify=False)
    energy = H.expectation(state).real
    variance = abs(H_state.norm_squared() - energy * energy)
    return energy, variance


def _state_deepcopy(state: MPS) -> MPS:
    data = [A.copy() for A in state]
    if isinstance(state, CanonicalMPS):
        return CanonicalMPS(
            data,
            center=state.center,
            normalize=False,
            strategy=state.strategy,
            is_canonical=True,
            error=state.error(),
        )
    return MPS(data, error=state.error())


def dmrg(
    H: MPO | NNHamiltonian,
    guess: MPS | None = None,
    maxiter: int = 20,
    tol: float = 1e-10,
    tol_up: float | None = None,
    tol_eigs: float | None = None,
    strategy: Strategy = DEFAULT_STRATEGY,
    callback: Callable | None = None,
) -> OptimizeResults:
    """Compute the ground state of a Hamiltonian represented as MPO using the
    two-site DMRG algorithm.

    Parameters
    ----------
    H : MPO | NNHamiltonian
        The Hermitian operator that is to be diagonalized. It may be also a
        nearest-neighbor Hamiltonian that is implicitly converted to MPO.
    guess : MPS | None
        An initial guess for the ground state.
    maxiter : int
        Maximum number of steps of the DMRG. Each step is a sweep that runs
        over every pair of neighborin sites. Defaults to 20.
    tol : float
        Tolerance in the energy to detect convergence of the algorithm.
    tol_up : float, default = `tol`
        If energy fluctuates up below this tolerance, continue the optimization.
    tol_eigs : float | None, default = `tol`
        Tolerance of Scipy's eigsh() solver, used internally. Zero means use
        machine precision.
    strategy : Strategy
        Truncation strategy to keep bond dimensions in check. Defaults to
        `DEFAULT_STRATEGY`, which is very strict.
    callback : Callable[[MPS, OptimizeResults], Any] | None
        A callable called after each iteration (defaults to None).

    Returns
    -------
    OptimizeResults
        The result from the algorithm in an :class:`~seemps.optimize.OptimizeResults`
        object.

    Examples
    --------
    >>> from seemps.hamiltonians import HeisenbergHamiltonian
    >>> from seemps.optimization import dmrg
    >>> H = HeisenbergHamiltonian(10)
    >>> result = dmrg(H)
    """
    if maxiter < 1:
        raise ValueError("maxiter must be positive")
    if isinstance(H, NNHamiltonian):
        H = H.to_mpo()
    if H.size < 2:
        raise ValueError("DMRG requires at least two sites")
    if guess is None:
        guess = random_mps(H.physical_dimensions(), D=2)
    if tol < 0:
        raise ValueError("tol must be non-negative")
    if tol_up is None:
        tol_up = tol
    elif tol_up < 0:
        raise ValueError("tol_up must be non-negative")
    if tol_eigs is None:
        tol_eigs = tol

    strat = strategy.replace(normalize=True)
    logger = make_logger()
    logger(f"DMRG initiated with maxiter={maxiter}, tolerance={tol}")

    if not isinstance(guess, CanonicalMPS) or guess.center not in (0, H.size - 1):
        guess = CanonicalMPS(guess, center=0, strategy=strat)
    guess.normalize_inplace()
    if guess.center == 0:
        direction = +1
        QF = QuadraticForm(H, guess, start=0)
    else:
        direction = -1
        QF = QuadraticForm(H, guess, start=H.size - 2)

    energy, variance = _energy_and_variance(H, QF.state)
    results = OptimizeResults(
        state=_state_deepcopy(QF.state),
        energy=energy,
        converged=False,
        message=f"Exceeded maximum number of steps {maxiter}",
        trajectory=[energy],
        variances=[variance],
    )
    logger(f"start, energy={energy}, variance={variance}")
    if callback is not None:
        callback(QF.state, results)

    last_energy = energy
    sweep = 0
    for sweep in range(1, maxiter + 1):
        local_energy = _sweep(QF, direction, tol_eigs, strat)
        direction = -direction

        energy, variance = _energy_and_variance(H, QF.state)
        results.trajectory.append(energy)
        results.variances.append(variance)
        logger(
            f"sweep={sweep}, eigenvalue={local_energy}, energy={energy}, variance={variance}"
        )
        if energy < results.energy:
            results.energy, results.state = energy, _state_deepcopy(QF.state)
        if callback is not None:
            callback(QF.state, results)

        energy_change = energy - last_energy
        energy_scale = max(1.0, abs(energy))
        reason = _convergence_reason(energy_change, energy_scale, tol, tol_up)
        if reason is not None:
            results.converged = True
            results.message = reason
            break
        last_energy = energy

    logger(
        f"DMRG finished with {sweep} sweeps:\nmessage = {results.message}\nconverged = {results.converged}"
    )
    logger.close()
    return results
