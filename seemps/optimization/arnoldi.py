import numpy as np
from numpy.typing import NDArray
from typing import Optional
from ..expectation import scprod
from ..state import MPS, CanonicalMPS, MPSSum, random_mps
from ..mpo import MPO, MPOList, MPOSum
from ..truncate.simplify import simplify
import scipy.linalg

from .descent import DESCENT_STRATEGY, OptimizeResults, Strategy


def _ill_conditioned_norm_matrix(N, tol=np.finfo(float).eps * 10):
    l = np.linalg.eigvalsh(N)[0]
    return np.any(np.abs(l) < tol)


class MPSArnoldiRepresentation:
    empty: NDArray = np.zeros((0, 0), dtype=np.complex128)
    operator: MPO
    H: NDArray
    N: NDArray
    V: list[MPS]
    strategy: Strategy

    def __init__(self, operator: MPO, strategy: Strategy = DESCENT_STRATEGY):
        self.operator = operator
        self.H = self.empty
        self.N = self.empty
        self.V = []
        self.strategy = strategy.replace(normalize=True)
        pass

    def add_vector(self, v: MPS) -> bool:
        v = simplify(v, strategy=self.strategy)
        new_H = np.pad(self.H + 0.0, ((0, 1), (0, 1)))
        new_N = np.pad(self.N + 0.0, ((0, 1), (0, 1)))
        n = [scprod(vi, v) for vi in self.V]
        new_N[:-1, -1] = n
        new_N[-1, :-1] = np.conj(n)
        new_N[-1, -1] = 1.0
        if len(new_N) > 1 and _ill_conditioned_norm_matrix(new_N):
            return v, False
        Op = self.operator
        h = [Op.expectation(vi, v) for vi in self.V]
        new_H[:-1, -1] = h
        new_H[-1, :-1] = np.conj(h)
        new_H[-1, -1] = Op.expectation(v).real
        self.H = new_H
        self.N = new_N
        self.V.append(v)
        return v, True

    def restart_with_ground_state(self) -> MPS:
        eigenvalues, eigenstates = scipy.linalg.eig(self.H, self.N)
        eigenvalues = eigenvalues.real
        ndx = np.argmin(eigenvalues)
        v = simplify(MPSSum(eigenstates[:, ndx], self.V), strategy=self.strategy)
        self.H = self.empty.copy()
        self.N = self.empty.copy()
        self.V = []
        v, _ = self.add_vector(v)
        return v, eigenvalues[ndx].real

    def variance_estimate(self) -> float:
        # Our basis is built as the sequence of normalized vectors
        # v, Hv/||Hv||, H^2v/||H^2v||, ...
        # This means
        # H[0,0] = <v|H|v>
        # H[0,1] = <v|H|Hv> / sqrt(<Hv|Hv>) = sqrt(<Hv|Hv>)
        return np.abs(self.H[0, 1]) ** 2 - np.abs(self.H[0, 0]) ** 2


def arnoldi_eigh(
    operator: MPO,
    v0: Optional[MPS] = None,
    maxiter: int = 100,
    nvectors: int = 10,
    tol: float = 1e-13,
    strategy: Strategy = DESCENT_STRATEGY,
    miniter: int = 1,
) -> OptimizeResults:
    if v0 is None:
        v0 = random_mps(operator.dimensions(), D=2)

    arnoldi = MPSArnoldiRepresentation(operator, strategy)
    arnoldi.add_vector(v0)
    v = operator @ v0
    best_energy = arnoldi.H[0, 0].real
    best_variance = abs(scprod(v, v)) - best_energy * best_energy
    best_vector = v0
    energies: list[float] = [best_energy]
    variances: list[float] = [best_variance]
    last_eigenvalue = variance = np.Inf
    message = f"Exceeded maximum number of steps {maxiter}"
    converged = True
    for i in range(maxiter):
        v, success = arnoldi.add_vector(v)
        if not success and nvectors == 2:
            message = "Unable to construct Arnoldi matrix"
            converged = False
            break
        L = len(arnoldi.V)
        if L == 2 and i > 0:
            if L > 1:
                variance = arnoldi.variance_estimate()
            variances.append(variance)
        elif L != 2 and i > 0:
            variances.append(variances[-1])
        if L == nvectors or not success:
            v, eigenvalue = arnoldi.restart_with_ground_state()
            eigenvalue_change, last_eigenvalue = (
                eigenvalue - last_eigenvalue,
                eigenvalue,
            )
            if np.abs(eigenvalue_change) < tol and i > miniter:
                message = f"Eigenvalue converged within tolerance {tol}"
                converged = True
                break
        v = operator @ v
        energy = arnoldi.H[0, 0].real
        energies.append(energy)
        if energy < best_energy:
            best_energy, best_vector, best_variance = energy, arnoldi.V[0], variance

    if converged:
        best_energy = operator.expectation(best_vector, best_vector).real
        energies.append(best_energy)
        v = operator @ best_vector
        variance = abs(scprod(v, v)) - best_energy * best_energy
        variances.append(variance)
    return OptimizeResults(
        state=best_vector,
        energy=best_energy,
        converged=converged,
        message=message,
        trajectory=energies,
        variances=variances,
    )
