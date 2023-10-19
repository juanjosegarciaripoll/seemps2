from dataclasses import dataclass
import numpy as np
from typing import Callable, List, Optional
from .skeleton import skeleton
from .mesh import Mesh
from ..state import MPS, random_mps


@dataclass
class Tracker:
    sweep: int = 0
    error: List[float] = []
    calls: List[int] = []
    times: List[float] = []
    maxrank: List[int] = []

    def set_error(self, error):
        self.error.append(error)

    def set_calls(self, calls):
        self.calls.append(calls)

    def set_time(self):
        pass

    def set_maxrank(self, mps: MPS):
        self.maxrank.append(max(mps.bond_dimensions()))


class Cross:
    def __init__(self, func: Callable, mesh: Mesh):
        self.func = func
        self.mesh = mesh
        self.qubits = [int(np.log2(s)) for s in mesh.shape()[:-1]]

    def initialize(self, mps: MPS, ordering: str = "A", **kwargs):
        # TODO: Check the mps against the mesh
        self.mps = mps
        self.ordering = ordering
        self.tracker = Tracker()
        self.structure = "binary" if np.all(mps.bond_dimensions() == 2) else "tt"
        self.I_physical = [
            np.arange(k, dtype=int).reshape(-1, 1)
            for k in self.mps.physical_dimensions()
        ]
        self.I_forward = [None for _ in range(len(self.mps) + 1)]
        self.I_backward = [None for _ in range(len(self.mps) + 1)]

        d = len(self.mps)
        # Forward pass
        R = np.ones((1, 1))
        for j in range(d):
            fiber = np.tensordot(R, self.mps[j], 1)
            self.mps[j], self.I_forward[j + 1], R = skeleton(
                fiber, self.I_physical[j], self.I_forward[j], ltr=True, **kwargs
            )
        self.mps[d - 1] = np.tensordot(self.mps[d - 1], R, 1)

        # Backward pass
        R = np.ones((1, 1))
        for j in range(d - 1, -1, -1):
            fiber = np.tensordot(self.mps[j], R, 1)
            self.mps[j], self.I_backward[j], R = skeleton(
                fiber, self.I_physical[j], self.I_backward[j + 1], ltr=True, **kwargs
            )
        self.mps[0] = np.tensordot(R, self.mps[0], 1)

    def sample(self, fiber: np.ndarray, ordering="A") -> np.ndarray:
        # Reshape the fiber as a tensor
        indices = None
        if self.structure == "binary":
            indices = binary_to_decimal(indices, self.qubits, self.ordering)
        return np.array([self.func(self.mesh[idx]) for idx in indices])

    def build_fiber(
        self, i_physical: np.ndarray, i_forward: np.ndarray, i_backward: np.ndarray
    ) -> np.ndarray:
        r1 = i_forward.shape[0] if i_forward is not None else 1
        s = i_physical.shape[0]
        r2 = i_backward.shape[0] if i_backward is not None else 1
        indices = np.kron(np.kron(_ones(r2), i_physical), _ones(r1))
        if i_forward is not None:
            indices = np.hstack((np.kron(_ones(s * r2), i_forward), indices))
        if i_backward is not None:
            indices = np.hstack((indices, np.kron(i_backward, _ones(r1 * s))))
        return indices.reshape((r1, s, r2), order="F")

    def measure(self, measurement_type: str = "sampling", **kwargs):
        pass

    @staticmethod
    def measure_sampling(mps: MPS, indices: np.ndarray) -> np.ndarray:
        pass

    @staticmethod
    def measure_norm(mps: MPS, norm_type: str = "norm2") -> float:
        pass

    @staticmethod
    def measure_integral(mps: MPS, integral_type: str = "simpson") -> float:
        pass

    def sweep(self, **kwargs):
        d = len(self.mps)

        # Forward pass
        R = np.ones((1, 1))
        for j in range(d):
            fiber = self.sample(
                self.I_physical[j], self.I_forward[j], self.I_backward[j + 1]
            )
            self.mps[j], self.I_forward[j + 1], R = skeleton(
                fiber, self.I_physical[j], self.I_forward[j], ltr=True, **kwargs
            )
        self.mps[d - 1] = np.tensordot(self.mps[d - 1], R, 1)

        # Backward pass
        R = np.ones((1, 1))
        for j in range(d - 1, -1, -1):
            fiber = self.sample(
                self.I_physical[j], self.I_forward[j], self.I_backward[j + 1]
            )
            self.mps[j], self.I_backward[j], R = skeleton(
                fiber, self.I_physical[j], self.I_backward[j + 1], ltr=False, **kwargs
            )
        self.mps[0] = np.tensordot(R, self.mps[0], 1)


def _ones(k, m=1):
    return np.ones((k, m), dtype=int)


# TODO: Clean and optimize
def binary_to_decimal(indices, qubits, ordering):
    """Maps an array of multi-indices in binary form to an array of
    multi-indices in decimal form."""

    def bitlist_to_int(bitlist):
        out = 0
        for bit in bitlist:
            out = (out << 1) | bit
        return out

    m = len(qubits)
    decimal_indices = []
    for idx, n in enumerate(qubits):
        if ordering == "A":
            rng = np.arange(idx * n, (idx + 1) * n)
        elif ordering == "B":
            rng = np.arange(idx, m * n, m)
        else:
            raise ValueError("Invalid ordering")
        decimal_ndx = bitlist_to_int(indices.T[rng])
        decimal_indices.append(decimal_ndx)

    decimal_indices = np.column_stack(decimal_indices)
    return decimal_indices


def cross_interpolation(
    func: Callable, mesh: Mesh, mps: Optional[MPS] = None, **kwargs
):
    cross = Cross(func, mesh)
    if mps is None:
        mps = random_mps([2] * sum(cross.qubits), 1)
    cross.initialize(mps, **kwargs)
    while not is_converged(cross.tracker, **kwargs):
        cross.sweep(**kwargs)
        cross.measure(**kwargs)

    return cross.mps, cross.tracker


def is_converged(
    tracker: Tracker,
    tol: float = 1e-10,
    maxcall: int = 10000000,
    maxiter: int = 1000,
    maxrank: int = 1000,
    maxtime: float = 60,
) -> bool:
    return False  # TODO: Implement
