from typing import Callable, Optional, Union, Optional

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

    def __init__(
        self,
        operator: MPO,
        strategy: Strategy = DESCENT_STRATEGY,
        tol_ill_conditioning: float = np.finfo(float).eps * 10,
    ):
        self.operator = operator
        self.H = self.empty
        self.N = self.empty
        self.V = []
        self.strategy = strategy.replace(normalize=True)
        self.tol_ill_conditioning = tol_ill_conditioning
        self._variance = None
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
        self._variance = None
        return v

    def restart_with_ground_state(self) -> tuple[MPS, float]:
        eigenvalues, eigenstates = scipy.linalg.eig(self.H, self.N)
        eigenvalues = eigenvalues.real
        ndx = np.argmin(eigenvalues)
        v = simplify(MPSSum(eigenstates[:, ndx], self.V), strategy=self.strategy)
        return self.restart_with_vector(v), eigenvalues[ndx].real

    def energy(self) -> float:
        return self.H[0, 0].real

    def variance(self) -> float:
        # Our basis is built as the sequence of normalized vectors
        # v, Hv/||Hv||, H^2v/||H^2v||, ...
        # This means
        # H[0,0] = <v|H|v>
        # H[0,1] = <v|H|Hv> / sqrt(<Hv|Hv>) = sqrt(<Hv|Hv>)
        if self._variance is None:
            energy = self.energy()
            H_v = self.operator.apply(self.V[0], strategy=NO_TRUNCATION)
            self._variance = abs(abs(scprod(H_v, H_v).real) - energy * energy)
        return self._variance

    def exponential(self, factor: Union[complex, float]) -> MPS:
        w = [0.0] * len(self.V)
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
                v, succeed = self.add_vector(v)
                if not succeed:
                    return False
        return True


def arnoldi_eigh(
    operator: MPO,
    v0: Optional[MPS] = None,
    maxiter: int = 100,
    nvectors: int = 10,
    tol: float = 1e-13,
    strategy: Strategy = DESCENT_STRATEGY,
    miniter: int = 1,
    callback: Optional[Callable] = None,
    tol_ill: float = np.finfo(float).eps * 10,
    **kwdargs,
) -> OptimizeResults:
    if v0 is None:
        v0 = random_mps(operator.dimensions(), D=2)
    arnoldi = MPSArnoldiRepresentation(operator, strategy, tol_ill)
    v: MPS = arnoldi.restart_with_vector(v0)
    results = OptimizeResults(
        state=v,
        energy=arnoldi.energy(),
        variances=[arnoldi.variance()],
        trajectory=[arnoldi.energy()],
        converged=False,
        message=f"Exceeded maximum number of steps {maxiter}",
    )
    if callback is not None:
        callback(arnoldi.V[0], results)
    last_energy = np.Inf
    for i in range(maxiter):
        v, success = arnoldi.add_vector(operator @ v)
        if not success and nvectors == 2:
            results.message = "Unable to construct Arnoldi matrix"
            results.converged = False
            break
        if len(arnoldi.V) == nvectors or not success:
            v, _ = arnoldi.restart_with_ground_state()
        energy = arnoldi.energy()
        results.trajectory.append(energy)
        results.variances.append(arnoldi.variance())
        if energy < results.energy:
            results.energy, results.state = energy, arnoldi.V[0]
        if callback is not None:
            callback(arnoldi.V[0], results)
        if len(arnoldi.V) == 1:
            energy_change = energy - last_energy
            if energy_change > abs(tol):
                results.message = f"Eigenvalue change {energy_change} fluctuates up above tolerance {tol_up}"
                results.converged = True
                break
            if (-abs(tol) <= energy_change) and i > miniter:
                results.message = f"Eigenvalue change below tolerance {tol}"
                results.converged = True
                break
            last_energy = energy
    return results
