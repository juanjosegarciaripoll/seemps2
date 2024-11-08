from __future__ import annotations
from typing import Callable, Optional, Union
import numpy as np
import scipy.sparse.linalg  # type: ignore
from opt_einsum import contract  # type: ignore
from ..tools import make_logger
from ..typing import Tensor4, Vector
from ..state import DEFAULT_STRATEGY, MPS, CanonicalMPS, Strategy, random_mps
from ..state._contractions import _contract_last_and_first
from ..state.environments import (
    MPOEnvironment,
    begin_mpo_environment,
    update_left_mpo_environment,
    update_right_mpo_environment,
)
from ..mpo import MPO
from ..hamiltonians import NNHamiltonian
from .descent import OptimizeResults


class QuadraticForm:
    H: MPO
    state: CanonicalMPS
    size: int
    left_env: list[MPOEnvironment]
    right_env: list[MPOEnvironment]
    site: int

    def __init__(self, H: MPO, state: CanonicalMPS, start: int = 0):
        self.H = H
        self.state = state
        self.size = size = state.size
        if size != H.size:
            raise Exception("In QuadraticForm, MPO and MPS do not have the same size")
        if any(O.shape[1] != A.shape[1] for O, A in zip(H, state)):
            raise Exception(
                "In QuadraticForm, MPO and MPS do not have matching dimensions"
            )
        left_env = [begin_mpo_environment()] * size
        right_env = left_env.copy()
        env = right_env[-1]
        for i in range(size - 1, start, -1):
            right_env[i - 1] = env = update_right_mpo_environment(
                env, state[i], H[i], state[i]
            )
        env = left_env[0]
        for i in range(0, start):
            left_env[i + 1] = env = update_left_mpo_environment(
                env, state[i], H[i], state[i]
            )
        self.left_env = left_env
        self.right_env = right_env
        self.site = start

    def two_site_Hamiltonian(self, i: int) -> scipy.sparse.linalg.LinearOperator:
        assert i == self.site
        L = self.left_env[i]
        R = self.right_env[i + 1]
        H12 = _contract_last_and_first(self.H[i], self.H[i + 1])
        c, i, k, j, l, d = H12.shape
        a, c, b = L.shape
        e, d, f = R.shape

        def perform_contraction(v: Vector) -> Vector:
            v = v.reshape(b, k, l, f)
            v = contract("acb,cikjld,edf,bklf->aije", L, H12, R, v)
            return v

        return scipy.sparse.linalg.LinearOperator(
            shape=(b * k * l * f, b * k * l * f),
            matvec=perform_contraction,
            dtype=type(L[0, 0, 0] * R[0, 0, 0] * H12[0, 0, 0, 0, 0, 0]),
        )

    def diagonalize(self, i: int, tol: float) -> tuple[float, Tensor4]:
        Op = self.two_site_Hamiltonian(i)
        v = _contract_last_and_first(self.state[i], self.state[i + 1])
        v /= np.linalg.norm(v.reshape(-1))
        eval, evec = scipy.sparse.linalg.eigsh(
            Op, 1, which="SA", v0=v.reshape(-1), tol=tol
        )
        return eval[0], evec.reshape(v.shape)

    def update_2site_right(self, AB: Tensor4, i: int, strategy: Strategy) -> None:
        self.state.update_2site_right(AB, i, strategy)
        if i < self.size - 2:
            self.site = j = i + 1
            self.left_env[j] = update_left_mpo_environment(
                self.left_env[i], self.state[i], self.H[i], self.state[i]
            )

    def update_2site_left(self, AB: Tensor4, i: int, strategy: Strategy) -> None:
        self.state.update_2site_left(AB, i, strategy)
        if i > 0:
            j = i + 1
            self.right_env[i] = update_right_mpo_environment(
                self.right_env[j], self.state[j], self.H[j], self.state[j]
            )
            self.site = i - 1


def dmrg(
    H: Union[MPO, NNHamiltonian],
    guess: Optional[MPS] = None,
    maxiter: int = 20,
    tol: float = 1e-10,
    tol_up: Optional[float] = None,
    tol_eigs: Optional[float] = None,
    strategy: Strategy = DEFAULT_STRATEGY,
    callback: Optional[Callable] = None,
) -> OptimizeResults:
    """Compute the ground state of a Hamiltonian represented as MPO using the
    two-site DMRG algorithm.

    Parameters
    ----------
    H : MPO | NNHamiltonian
        The Hermitian operator that is to be diagonalized. It may be also a
        nearest-neighbor Hamiltonian that is implicitly converted to MPO.
    guess : Optional[MPS]
        An initial guess for the ground state.
    maxiter : int
        Maximum number of steps of the DMRG. Each step is a sweep that runs
        over every pair of neighborin sites. Defaults to 20.
    tol : float
        Tolerance in the energy to detect convergence of the algorithm.
    tol_up : float, default = `tol`
        If energy fluctuates up below this tolerance, continue the optimization.
    tol_eigs : Optional[float], default = `tol`
        Tolerance of Scipy's eigsh() solver, used internally. Zero means use
        machine precision.
    strategy : Strategy
        Truncation strategy to keep bond dimensions in check. Defaults to
        `DEFAULT_STRATEGY`, which is very strict.
    callback : Optional[Callable[[MPS, OptimizeResults], Any]]
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
    if guess.center == 0:
        direction = +1
        QF = QuadraticForm(H, guess, start=0)
    else:
        direction = -1
        QF = QuadraticForm(H, guess, start=H.size - 2)
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
    for step in range(maxiter):
        if direction > 0:
            for i in range(0, H.size - 1):
                E, AB = QF.diagonalize(i, tol=tol_eigs)
                QF.update_2site_right(AB, i, strategy)
                logger(f"-> site={i}, eigenvalue={E}")
        else:
            for i in range(H.size - 2, -1, -1):
                E, AB = QF.diagonalize(i, tol=tol_eigs)
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
        f"DMRG finished with {step} iterations:\n"
        f"message = {results.message}\nconverged = {results.converged}"
    )
    logger.close()
    return results
