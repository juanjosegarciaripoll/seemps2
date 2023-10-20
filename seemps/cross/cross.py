from dataclasses import dataclass, field, replace
import numpy as np
from typing import Callable, List, Optional
from .maxvol import maxvol_sqr, maxvol_rct
from .mesh import Mesh
from ..state import MPS, random_mps


@dataclass
class CrossStrategy:
    mps_ordering: str = "A"
    maxvol_sqr_tau: float = 1.05
    maxvol_sqr_maxiter: int = 100
    maxvol_rct_tau: float = 1.10
    maxvol_rct_minrank: int = 1
    maxvol_rct_maxrank: int = 1
    measurement_type: str = "sampling"
    sampling_points: int = 1000
    verbose: bool = True


@dataclass
class CrossConvergence:
    tol: float = 1e-10
    maxiter: int = 1000
    maxcall: int = 10000000
    maxrank: int = 1000


@dataclass
class Tracker:
    sweep: int = 0
    error: List[float] = field(default_factory=lambda: [1])
    calls: List[int] = field(default_factory=lambda: [0])
    maxrank: List[int] = field(default_factory=lambda: [0])

    def set_error(self, error):
        self.error.append(error)

    def set_calls(self, calls):
        self.calls.append(calls)

    def set_maxrank(self, mps: MPS):
        self.maxrank.append(max(mps.bond_dimensions()))


class Cross:
    def __init__(
        self,
        func: Callable,
        mesh: Mesh,
        mps: MPS,
        cross_strategy: CrossStrategy,
        tracker: Tracker = Tracker(),
    ):
        self.func = func
        self.mesh = mesh
        self.mps = mps
        self.cross_strategy = cross_strategy
        self.tracker = tracker

        # TODO: Check if the mps is compatible with the mesh
        self.sites = len(self.mps)
        self.qubits = [int(np.log2(s)) for s in mesh.shape()[:-1]]
        self.structure = (
            "binary" if all(dim == 2 for dim in mps.physical_dimensions()) else "tt"
        )
        self.presweep()

    def sample(self, indices: np.ndarray) -> np.ndarray:
        if self.structure == "binary":
            indices = self.binary_to_decimal(indices)
        return np.array([self.func(self.mesh[idx]) for idx in indices])

    def evaluate(self, j: int) -> np.ndarray:
        i_physical = self.I_physical[j]
        i_forward = self.I_forward[j]
        i_backward = self.I_backward[j + 1]
        r1 = i_forward.shape[0] if i_forward is not None else 1
        s = i_physical.shape[0]
        r2 = i_backward.shape[0] if i_backward is not None else 1
        indices = np.kron(
            np.kron(np.ones((r2, 1), dtype=int), i_physical),
            np.ones((r1, 1), dtype=int),
        )
        if i_forward is not None:
            indices = np.hstack(
                (np.kron(np.ones((s * r2, 1), dtype=int), i_forward), indices)
            )
        if i_backward is not None:
            indices = np.hstack(
                (indices, np.kron(i_backward, np.ones((r1 * s, 1), dtype=int)))
            )
        if len(self.tracker.calls) < self.tracker.sweep + 1:
            self.tracker.set_calls(0)
        self.tracker.calls[self.tracker.sweep] += len(indices)
        fiber = self.sample(indices)
        return fiber.reshape((r1, s, r2), order="F")

    def skeleton(self, fiber: np.ndarray, j: int, ltr: bool):
        r1, s, r2 = fiber.shape

        kwargs_sqr = {
            "k": self.cross_strategy.maxvol_sqr_maxiter,
            "e": self.cross_strategy.maxvol_sqr_tau,
        }
        kwargs_rct = {
            "tau": self.cross_strategy.maxvol_rct_tau,
            "min_r": self.cross_strategy.maxvol_rct_minrank,
            "max_r": self.cross_strategy.maxvol_rct_maxrank,
        }
        kwargs_rct.update(kwargs_sqr)
        if self.tracker.sweep == 0:
            kwargs_rct["min_r"] = 0
            kwargs_rct["max_r"] = 0

        if ltr:
            i_virtual = self.I_forward[j]
            fiber = np.reshape(fiber, (r1 * s, r2), order="F")
        else:
            i_virtual = self.I_backward[j + 1]
            fiber = np.reshape(fiber, (r1, s * r2), order="F").T

        Q, R = np.linalg.qr(fiber)
        if Q.shape[0] <= Q.shape[1]:
            i_maxvol = np.arange(Q.shape[0], dtype=int)
            Q_maxvol = np.eye(Q.shape[0], dtype=float)
        elif kwargs_rct["max_r"] == 0:
            i_maxvol, Q_maxvol = maxvol_sqr(Q, **kwargs_sqr)
        else:
            i_maxvol, Q_maxvol = maxvol_rct(Q, **kwargs_rct)

        i_physical = self.I_physical[j]
        if ltr:
            i_physical_ext = np.kron(i_physical, np.ones((r1, 1), dtype=int))
            fiber = np.reshape(Q_maxvol, (r1, s, -1), order="F")
            R = Q[i_maxvol, :] @ R
        else:
            i_physical_ext = np.kron(np.ones((r2, 1), dtype=int), i_physical)
            fiber = np.reshape(Q_maxvol.T, (-1, s, r2), order="F")
            R = (Q[i_maxvol, :] @ R).T

        if i_virtual is not None:
            i_virtual_ext = (
                np.kron(np.ones((s, 1), dtype=int), i_virtual)
                if ltr
                else np.kron(i_virtual, np.ones((s, 1), dtype=int))
            )
            i_physical_ext = (
                np.hstack((i_virtual_ext, i_physical_ext))
                if ltr
                else np.hstack((i_physical_ext, i_virtual_ext))
            )
        i_maxvol = i_physical_ext[i_maxvol, :]
        return fiber, i_maxvol, R

    def measure(self):
        if self.cross_strategy.measurement_type == "sampling":
            error = self.measure_sampling()
            log_name = "Sampling"
        # TODO: Implement
        # elif measurement_type == "norm":
        #     error = self.measure_norm(**kwargs)
        #     log_name = "Norm"
        # elif measurement_type == "integral":
        #     error = self.measure_integral(**kwargs)
        #     log_name = "Integral"
        else:
            raise ValueError("Invalid measurement_type")

        self.tracker.set_error(error)

        # TODO: Implement as a log using the seemps debug parameter
        if self.cross_strategy.verbose:
            print(
                f"Sweep {self.tracker.sweep:<3} | "
                + f"Max χ {self.tracker.maxrank[-1]:>3} | "
                + f"{log_name} {error:.2E} | "
                + f"Function calls {self.tracker.calls[-1]:>8}"
            )

    def measure_sampling(self) -> float:
        if self.tracker.sweep == 1:
            self.tracker.sampling_indices = np.vstack(
                [
                    np.random.choice(k, self.cross_strategy.sampling_points)
                    for k in self.mps.physical_dimensions()
                ]
            ).T
            self.tracker.samples = self.sample(self.tracker.sampling_indices)
        Q = self.mps[0][0, self.tracker.sampling_indices[:, 0], :]
        for i in range(1, self.sites):
            Q = np.einsum(
                "kq,qkr->kr", Q, self.mps[i][:, self.tracker.sampling_indices[:, i], :]
            )
        error = np.linalg.norm(Q[:, 0] - self.tracker.samples) / np.linalg.norm(
            self.tracker.samples
        )

        return error

    # def measure_norm(self, norm_type: str = "norm2") -> float:
    #     return error

    # def measure_integral(self, integral_type: str = "simpson") -> float:
    #     return error

    def presweep(self):
        self.I_physical = [
            np.arange(k, dtype=int).reshape(-1, 1)
            for k in self.mps.physical_dimensions()
        ]
        self.I_forward = [None for _ in range(self.sites + 1)]
        self.I_backward = [None for _ in range(self.sites + 1)]

        # Forward pass
        R = np.ones((1, 1))
        for j in range(self.sites):
            fiber = np.tensordot(R, self.mps[j], 1)
            self.mps[j], self.I_forward[j + 1], R = self.skeleton(fiber, j, ltr=True)
        self.mps[self.sites - 1] = np.tensordot(self.mps[self.sites - 1], R, 1)

        # Backward pass
        R = np.ones((1, 1))
        for j in range(self.sites - 1, -1, -1):
            fiber = np.tensordot(self.mps[j], R, 1)
            self.mps[j], self.I_backward[j], R = self.skeleton(fiber, j, ltr=False)
        self.mps[0] = np.tensordot(R, self.mps[0], 1)

    def sweep(self):
        # Forward pass
        R = np.ones((1, 1))
        for j in range(self.sites):
            fiber = self.evaluate(j)
            self.mps[j], self.I_forward[j + 1], R = self.skeleton(fiber, j, ltr=True)
        self.mps[self.sites - 1] = np.tensordot(self.mps[self.sites - 1], R, 1)

        # Backward pass
        R = np.ones((1, 1))
        for j in range(self.sites - 1, -1, -1):
            fiber = self.evaluate(j)
            self.mps[j], self.I_backward[j], R = self.skeleton(fiber, j, ltr=False)
        self.mps[0] = np.tensordot(R, self.mps[0], 1)

        self.tracker.sweep += 1
        self.tracker.set_maxrank(self.mps)

    def binary_to_decimal(self, indices):
        def bitlist_to_int(bitlist):
            out = 0
            for bit in bitlist:
                out = (out << 1) | bit
            return out

        m = len(self.qubits)
        decimal_indices = []
        for idx, n in enumerate(self.qubits):
            if self.cross_strategy.mps_ordering == "A":
                rng = np.arange(idx * n, (idx + 1) * n)
            elif self.cross_strategy.mps_ordering == "B":
                rng = np.arange(idx, m * n, m)
            else:
                raise ValueError("Invalid ordering")
            decimal_ndx = bitlist_to_int(indices.T[rng])
            decimal_indices.append(decimal_ndx)
        decimal_indices = np.column_stack(decimal_indices)
        return decimal_indices


def cross_interpolation(
    func: Callable,
    mesh: Mesh,
    mps0: Optional[MPS] = None,
    cross_strategy: CrossStrategy = CrossStrategy(),
    cross_convergence: CrossConvergence = CrossConvergence(),
):
    if mps0 is None:
        qubits = [int(np.log2(s)) for s in mesh.shape()[:-1]]
        mps0 = random_mps([2] * sum(qubits), 1, rng=np.random.default_rng(42))
    cross = Cross(func, mesh, mps0, cross_strategy)
    while not converged(cross.tracker, cross_convergence):
        cross.sweep()
        cross.measure()
    return cross.mps, cross.tracker


def converged(tracker: Tracker, cross_convergence: CrossConvergence) -> bool:
    return (
        tracker.error[-1] < cross_convergence.tol
        or tracker.sweep >= cross_convergence.maxiter
        or tracker.calls[-1] >= cross_convergence.maxcall
        or tracker.maxrank[-1] >= cross_convergence.maxrank
    )
