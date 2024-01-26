from __future__ import annotations
from typing import Callable, Optional, Union, Optional, Any

import numpy as np
import scipy.linalg  # type: ignore
from numpy.typing import NDArray

from ..expectation import scprod
from ..mpo import MPO
from ..state import MPS, CanonicalMPS, MPSSum, random_mps, Strategy, NO_TRUNCATION
from ..truncate.simplify import simplify
from .descent import DESCENT_STRATEGY, OptimizeResults


class MPSArnoldiRepresentation:
    empty: NDArray = np.zeros((0, 0))
    operator: MPO
    H: NDArray
    N: NDArray
    V: list[MPS]
    strategy: Strategy
    tol_ill_conditioning: float
    _variance: Optional[float]
    gamma: float = 0.0
    _eigenvector: NDArray
    _energy: Optional[float]
    _variance: Optional[float]

    def __init__(
        self,
        operator: MPO,
        strategy: Strategy = DESCENT_STRATEGY,
        tol_ill_conditioning: float = np.finfo(float).eps * 10,
        gamma: float = 0.0,
    ):
        self.operator = operator
        self.H = self.empty
        self.N = self.empty
        self.V = []
        self.strategy = strategy.replace(normalize=True)
        self.tol_ill_conditioning = tol_ill_conditioning
        self._eigenvector = None
        self._variance = None
        self._energy = None
        self.gamma = gamma
        pass

    def _ill_conditioned_norm_matrix(self, N: np.ndarray) -> bool:
        l = np.linalg.eigvalsh(N)[0]
        return np.any(np.abs(l) < self.tol_ill_conditioning)

    def add_vector(self, v: MPS) -> tuple[MPS, bool]:
        # We no longer should need this. Restart takes care of creating
        # a simplified vector, and the user is responsible for letting
        # the MPO do something sensible.
        if isinstance(v, CanonicalMPS):
            v.normalize_inplace()
        else:
            v = simplify(v, strategy=self.strategy)
        n = np.asarray([scprod(vi, v) for vi in self.V]).reshape(-1, 1)
        new_N = np.block([[self.N, n], [n.T.conj(), 1.0]])
        if len(new_N) > 1 and self._ill_conditioned_norm_matrix(new_N):
            return v, False
        self.N = new_N
        Op = self.operator
        h = np.asarray([Op.expectation(vi, v) for vi in self.V]).reshape(-1, 1)
        self.H = np.block([[self.H, h], [h.T.conj(), Op.expectation(v).real]])
        self.V.append(v)
        return v, True

    def restart_with_vector(self, v: MPS) -> MPS:
        self.H = self.empty.copy()
        self.N = self.empty.copy()
        self.V = []
        v, _ = self.add_vector(v)
        return v

    def restart_with_ground_state(self) -> tuple[MPS, float]:
        eigenvalues, eigenstates = scipy.linalg.eig(self.H, self.N)
        eigenvalues = eigenvalues.real
        ndx = np.argmin(eigenvalues)
        v = simplify(MPSSum(eigenstates[:, ndx], self.V), strategy=self.strategy)
        if self.gamma == 0 or self._eigenvector is None:
            new_v = v
        else:
            print(f"gamma = {self.gamma}")
            new_v = simplify(
                MPSSum([1 - self.gamma, self.gamma], [v, self._eigenvector]),
                strategy=self.strategy,
            )
        new_v = self.restart_with_vector(new_v)
        self._eigenvector = v
        self._variance = self._energy = None
        return new_v

    def eigenvector(self) -> MPS:
        return self.V[0] if self._eigenvector is None else self._eigenvector

    def energy_and_variance(self) -> float:
        # Our basis is built as the sequence of normalized vectors
        # v, Hv/||Hv||, H^2v/||H^2v||, ...
        # This means
        # H[0,0] = <v|H|v>
        # H[0,1] = <v|H|Hv> / sqrt(<Hv|Hv>) = sqrt(<Hv|Hv>)
        if self._energy is None or self._variance is None:
            v = self.eigenvector()
            self._energy = energy = self.operator.expectation(v)
            H_v = self.operator.apply(self.V[0], strategy=NO_TRUNCATION)
            self._variance = abs(abs(scprod(H_v, H_v).real) - energy * energy)
        return self._energy, self._variance

    def exponential(self, factor: Union[complex, float]) -> MPS:
        w = np.zeros(self.V)
        w[0] = 1.0
        w = scipy.sparse.linalg.expm_multiply(factor * self.H, w)
        return simplify(MPSSum(w, self.V), strategy=self.strategy)

    def build_Krylov_basis(self, v: MPS, order: int) -> bool:
        """Build a Krylov basis up to given order. Returns False
        if the size of the basis had to be truncated due to linear
        dependencies between vectors."""
        for i in range(order):
            if i == 0:
                v = self.restart_with_vector(v)
            else:
                v, succeed = self.add_vector(self.operator @ v)
                if not succeed:
                    return False
        return True


def arnoldi_eigh(
    operator: MPO,
    guess: Optional[MPS] = None,
    maxiter: int = 100,
    nvectors: int = 10,
    tol: float = 1e-13,
    tol_ill: float = np.finfo(float).eps * 10,
    tol_up: Optional[float] = None,
    gamma: float = 0,
    strategy: Strategy = DESCENT_STRATEGY,
    callback: Optional[Callable[[MPS, OptimizeResults], Any]] = None,
) -> OptimizeResults:
    """Ground state search of Hamiltonian `H` by the Arnoldi method.

    Parameters
    ----------
    H : Union[MPO, MPOList, MPOSum]
        Hamiltonian in MPO form.
    guess : Optional[MPS]
        Initial guess of the ground state. If None, defaults to a random
        MPS deduced from the operator's dimensions.
    maxiter : int
        Maximum number of iterations (defaults to 1000).
    nvectors: int
        Number of vectors in the Krylov basis (defaults to 10).
    tol : float
        Energy variation that indicates termination (defaults to 1e-13).
    tol_up : float, default = `tol`
        If energy fluctuates up below this tolerance, continue the optimization.
    tol_ill : float
        Check for ill conditioning of the Krylov basis (defaults to 1e-15).
    gamma : float
        If nonzero, convergence acceleration factor. Default is 0.0 (no inertia).
        Alternatively, provide -0.75.
    strategy : Optional[Strategy]
        Linear combination of MPS truncation strategy. Defaults to
        DESCENT_STRATEGY.
    callback : Optional[Callable[[MPS, OptimizeResults], Any]]
        A callable called after each iteration (defaults to None).

    Results
    -------
    OptimizeResults
        Results from the optimization. See :class:`OptimizeResults`.
    """
    if guess is None:
        guess = random_mps(operator.dimensions(), D=2)
    if tol_up is None:
        tol_up = abs(tol)
    arnoldi = MPSArnoldiRepresentation(
        operator, strategy, tol_ill_conditioning=tol_ill, gamma=gamma
    )
    v: MPS = arnoldi.restart_with_vector(guess)
    energy, variance = arnoldi.energy_and_variance()
    results = OptimizeResults(
        state=v,
        energy=energy,
        variances=[variance],
        trajectory=[energy],
        converged=False,
        message=f"Exceeded maximum number of steps {maxiter}",
    )
    if callback is not None:
        callback(arnoldi.V[0], results)
    last_energy = energy
    for i in range(maxiter):
        v, success = arnoldi.add_vector(operator @ v)
        if not success and nvectors == 2:
            results.message = "Unable to construct Arnoldi matrix"
            results.converged = False
            break
        if len(arnoldi.V) == nvectors or not success:
            v = arnoldi.restart_with_ground_state()
        energy, variance = arnoldi.energy_and_variance()
        results.trajectory.append(energy)
        results.variances.append(variance)
        if energy < results.energy:
            results.energy, results.state = energy, arnoldi.eigenvector()
        if callback is not None:
            callback(arnoldi.eigenvector(), results)
        if len(arnoldi.V) == 1:
            energy_change = energy - last_energy
            if energy_change > abs(tol_up * energy):
                results.message = f"Eigenvalue change {energy_change} fluctuates up above tolerance {tol_up}"
                results.converged = True
                break
            if -abs(tol * energy) <= energy_change:
                results.message = f"Eigenvalue change below tolerance {tol}"
                results.converged = True
                break
            last_energy = energy
    return results
