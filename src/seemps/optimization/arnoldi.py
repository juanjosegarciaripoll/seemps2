from __future__ import annotations
from typing import Callable, Optional, Union, Any

import numpy as np
import scipy.linalg  # type: ignore
from numpy.typing import NDArray

from ..expectation import scprod
from ..operators import MPO, MPOSum, MPOList
from ..state import MPS, CanonicalMPS, MPSSum, random_mps
from ..truncate.simplify import simplify
from .descent import DESCENT_STRATEGY, OptimizeResults, Strategy


def _ill_conditioned_norm_matrix(N, tol=np.finfo(float).eps * 10):
    l = np.linalg.eigvalsh(N)[0]
    return np.any(np.abs(l) < tol)


class MPSArnoldiRepresentation:
    empty: NDArray = np.zeros((0, 0))
    operator: Union[MPO, MPOSum]
    H: NDArray
    N: NDArray
    V: list[MPS]
    strategy: Strategy

    def __init__(
        self,
        operator: Union[MPO, MPOSum],
        strategy: Strategy = DESCENT_STRATEGY,
    ):
        self.operator = operator
        self.H = self.empty
        self.N = self.empty
        self.V = []
        self.strategy = strategy.replace(normalize=True)
        pass

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
        if len(new_N) > 1 and _ill_conditioned_norm_matrix(new_N):
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
        return self.restart_with_vector(v), eigenvalues[ndx].real

    def variance_estimate(self) -> float:
        # Our basis is built as the sequence of normalized vectors
        # v, Hv/||Hv||, H^2v/||H^2v||, ...
        # This means
        # H[0,0] = <v|H|v>
        # H[0,1] = <v|H|Hv> / sqrt(<Hv|Hv>) = sqrt(<Hv|Hv>)
        return abs(np.abs(self.H[0, 1]) ** 2 - np.abs(self.H[0, 0]) ** 2)

    def exponential(self, factor: Union[complex, float]) -> MPS:
        w = np.zeros(len(self.V))
        w[0] = 1.0
        w = scipy.sparse.linalg.expm_multiply(factor * self.H, w)
        return simplify(MPSSum(w.tolist(), self.V), strategy=self.strategy)

    def build_Krylov_basis(self, v: MPS, order: int) -> bool:
        """Build a Krylov basis up to given order. Returns False
        if the size of the basis had to be truncated due to linear
        dependencies between vectors."""
        for i in range(order):
            if i == 0:
                v = self.restart_with_vector(v)
            else:
                v, succeed = self.add_vector(v)
                if not succeed:
                    return False
        return True


def arnoldi_eigh(
    operator: Union[MPO, MPOSum],
    v0: Optional[MPS] = None,
    nvectors: int = 10,
    maxiter: int = 100,
    tol: float = 1e-13,
    strategy: Strategy = DESCENT_STRATEGY,
    miniter: int = 1,
    callback: Optional[Callable[[MPS, float, OptimizeResults], Any]] = None,
) -> OptimizeResults:
    """Ground state search of Hamiltonian `H` by Arnoldi diagonalization.

    Parameters
    ----------
    H : Union[MPO, MPOSum]
        Hamiltonian in MPO form.
    state : MPS
        Initial guess of the ground state.
    nvectors : int
        Number of vectors in the Arnoldi algorithm (defaults to 10).
        It must be an integer between 2 and 100, both included.
    maxiter : int
        Maximum number of iterations (defaults to 1000).
    tol : float
        Energy variation that indicates termination (defaults to 1e-13).
    strategy : Optional[Strategy]
        Linear combination of MPS truncation strategy. Defaults to
        DESCENT_STRATEGY.
    callback : Optional[Callable[[MPS, float, OptimizeResults], Any]]
        A callable called after each iteration with the current state,
        an estimate of the energy, and the accumulated results object.
        Defaults to None.

    Results
    -------
    OptimizeResults
        Results from the optimization. See :class:`OptimizeResults`.
    """
    if v0 is None:
        v0 = random_mps(operator.dimensions(), D=2)
    if nvectors < 2 or nvectors > 100 or not isinstance(nvectors, int):
        raise ValueError("nvectors must be an integer between 2 and 100")
    arnoldi = MPSArnoldiRepresentation(operator, strategy)
    arnoldi.add_vector(v0)
    v: MPS = operator @ v0  # type: ignore
    energy = arnoldi.H[0, 0].real
    variance = abs(scprod(v, v) - energy * energy)
    results = OptimizeResults(
        state=v0,
        energy=energy,
        trajectory=[energy],
        variances=[variance],
        message=f"Exceeded maximum number of steps {maxiter}",
        converged=False,
    )
    if callback is not None:
        callback(arnoldi.V[0], energy, results)
    last_eigenvalue = np.Inf
    for i in range(maxiter):
        v, success = arnoldi.add_vector(v)
        if not success and nvectors == 2:
            results.message = "Unable to construct Arnoldi matrix"
            results.converged = False
            break
        L = len(arnoldi.V)
        if L > 1:
            variance = arnoldi.variance_estimate()
        results.variances.append(variance)
        if L == nvectors or not success:
            v, eigenvalue = arnoldi.restart_with_ground_state()
            eigenvalue_change, last_eigenvalue = (
                eigenvalue - last_eigenvalue,
                eigenvalue,
            )
            if (eigenvalue_change >= -abs(tol)) and i > miniter:
                results.message = f"Eigenvalue converged within tolerance {tol}"
                results.converged = True
                break
        energy = arnoldi.H[0, 0].real
        results.trajectory.append(energy)
        if energy < results.energy:
            results.energy, results.state = energy, arnoldi.V[0]
        if callback is not None:
            callback(arnoldi.V[0], energy, results)
        v = operator @ v  # type: ignore
    return results
