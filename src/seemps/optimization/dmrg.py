from typing import Callable

import numpy as np
import scipy.sparse.linalg  # type: ignore
from opt_einsum import contract  # type: ignore

from ..hamiltonians import NNHamiltonian
from ..mpo import MPO
from ..state import DEFAULT_STRATEGY, MPS, CanonicalMPS, Strategy, random_mps
from ..state._contractions import _contract_last_and_first
from ..state.environments import (
    MPOEnvironment,
    begin_mpo_environment,
    update_left_mpo_environment,
    update_right_mpo_environment,
)
from ..tools import log
from ..typing import Optional, Tensor4, Union, Vector
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

    def diagonalize(self, i: int) -> tuple[float, Tensor4]:
        Op = self.two_site_Hamiltonian(i)
        v = _contract_last_and_first(self.state[i], self.state[i + 1])
        eval, evec = scipy.sparse.linalg.eigsh(Op, 1, which="SA", v0=v.reshape(-1))
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
    strategy: Strategy = DEFAULT_STRATEGY,
    tol: float = 1e-10,
    maxiter: int = 20,
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
    strategy : Strategy
        Truncation strategy to keep bond dimensions in check. Defaults to
        `DEFAULT_STRATEGY`, which is very strict.
    tol : float
        Tolerance in the energy to detect convergence of the algorithm.
    maxiter : int
        Maximum number of steps of the DMRG. Each step is a sweep that runs
        over every pair of neighborin sites. Defaults to 20.
    callback : Optional[callable]
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

    if not isinstance(guess, CanonicalMPS):
        guess = CanonicalMPS(guess, center=0)
    if guess.center == 0:
        direction = +1
        QF = QuadraticForm(H, guess, start=0)
    else:
        direction = -1
        QF = QuadraticForm(H, guess, start=H.size - 2)
    best_energy = np.Inf
    best_vector = guess
    oldE = np.inf
    energies = []
    converged = False
    msg = "DMRG did not converge"
    strategy = strategy.replace(normalize=True)
    for step in range(maxiter):
        if direction > 0:
            for i in range(0, H.size - 1):
                newE, AB = QF.diagonalize(i)
                QF.update_2site_right(AB, i, strategy)
                log(f"-> site={i}, energy={newE}, {H.expectation(QF.state)}")
        else:
            for i in range(H.size - 2, -1, -1):
                newE, AB = QF.diagonalize(i)
                QF.update_2site_left(AB, i, strategy)
                log(f"<- site={i}, energy={newE}, {H.expectation(QF.state)}")

        if callback is not None:
<<<<<<< HEAD
            callback(QF.state)
=======
            callback(QF.state, newE)
>>>>>>> efa1ec87dedd2493560275876751c3699594fafd
        log(
            f"step={step}, energy={newE}, change={oldE-newE}, {H.expectation(QF.state)}"
        )
        energies.append(newE)
        if newE < best_energy:
            best_energy, best_vector = newE, QF.state
        if newE - oldE >= abs(tol) or newE - oldE >= -abs(
            tol
        ):  # This criteria makes it stop
            msg = "Energy change below tolerance"
            log(msg)
            converged = True
            break
        direction = -direction
        oldE = newE
    if not converged:
        guess = CanonicalMPS(QF.state, center=0, normalize=True)
        newE = H.expectation(guess).real
        energies.append(newE)
        if newE < best_energy:
            best_energy, best_vector = newE, QF.state
        best_vector = CanonicalMPS(best_vector, center=0, normalize=True)
    return OptimizeResults(
        state=best_vector,
        energy=best_energy,
        converged=converged,
        message=msg,
        trajectory=energies,
    )
