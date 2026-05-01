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
    QF: QuadraticForm, i: int, tol: float
) -> tuple[float, Tensor4]:
    Op = QF.two_site_operator(i)
    v = _contract_last_and_first(QF.state[i], QF.state[i + 1])
    v_shape = v.shape
    v = v.reshape(-1)
    v /= np.linalg.norm(v)
    eval, evec = scipy.sparse.linalg.eigsh(
        Op, 1, which="SA", v0=v, tol=tol
    )
    return eval[0], evec.reshape(v_shape)


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
        raise Exception("maxiter cannot be zero or negative")
    if isinstance(H, NNHamiltonian):
        H = H.to_mpo()
    if guess is None:
        guess = random_mps(H.dimensions(), D=2)
    if tol_up is None:
        tol_up = abs(tol)
    if tol_eigs is None:
        tol_eigs = tol

    logger = make_logger()
    logger(f"DMRG initiated with maxiter={maxiter}, relative tolerance={tol}")
    if not isinstance(guess, CanonicalMPS):
        guess = CanonicalMPS(guess, center=0)
    if guess.center <= guess.size // 2:
        direction = +1
        guess = CanonicalMPS(guess, center=0)
        QF = QuadraticForm(H, guess, start=0)
    else:
        direction = -1
        guess = CanonicalMPS(guess, center=-1)
        QF = QuadraticForm(H, guess, start=guess.size - 2)
    energy = H.expectation(QF.state).real
    variance = abs(H.apply(QF.state).norm_squared() - energy * energy)
    results = OptimizeResults(
        state=QF.state.copy(),
        energy=energy,
        converged=False,
        message=f"Exceeded maximum number of steps {maxiter}",
        trajectory=[energy],
        variances=[variance],
    )
    logger(f"start, energy={energy}, variance={variance}")
    if callback is not None:
        callback(QF.state, results)
    E: float = np.inf
    last_E: float = E
    strategy = strategy.replace(normalize=True)
    step: int = 0
    for step in range(maxiter):
        if direction > 0:
            for i in range(0, H.size - 1):
                E, AB = _diagonalize_two_site(QF, i, tol_eigs)
                QF.update_2site_right(AB, i, strategy)
                logger(f"-> site={i}, eigenvalue={E}")
        else:
            for i in range(H.size - 2, -1, -1):
                E, AB = _diagonalize_two_site(QF, i, tol_eigs)
                QF.update_2site_left(AB, i, strategy)
                logger(f"<- site={i}, eigenvalue={E}")

        # In principle, E is the exact eigenvalue. However, we have
        # truncated the eigenvector, which means that the computation of
        # the residual cannot use that value
        energy = H.expectation(QF.state).real
        H_state = H @ QF.state
        variance = abs(H_state.norm_squared() - energy * energy)

        results.trajectory.append(E)
        results.variances.append(variance)
        logger(f"step={step}, eigenvalue={E}, energy={energy}, variance={variance}")
        if E < results.energy:
            results.energy, results.state = E, QF.state.copy()
        if callback is not None:
            callback(QF.state, results)

        energy_change = E - last_E
        if energy_change > abs(tol_up * E):
            results.message = f"Energy fluctuation above tolerance {tol_up}"
            results.converged = True
            break
        if -abs(tol * E) <= energy_change <= 0:
            results.message = f"Energy decrease slower than tolerance {tol}"
            results.converged = True
            break
        direction = -direction
        last_E = E
    logger(
        f"DMRG finished with {step + 1} iterations:\nmessage = {results.message}\nconverged = {results.converged}"
    )
    logger.close()
    return results
