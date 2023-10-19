from dataclasses import dataclass, replace
import numpy as np
from typing import Callable, List, Optional
from .skeleton import skeleton
from .mesh import Mesh
from ..state import MPS, random_mps


@dataclass
class CrossOptions:
    # Convergence options, arguments of is_converged()
    tol: float = 1e-10
    maxcall: int = 10000000
    maxiter: int = 1000
    maxrank: int = 1000
    maxtime: float = 60
    maxvol_minrank: int = 0
    maxvol_maxrank: int = 1
    maxvol_sqr_tau: float = 1.05
    maxvol_sqr_maxiter: int = 100
    maxvol_rct_tau: float = 1.1
    maxvol_rct_maxiter: int = 100
    # Measurement options, arguments of measure().
    measurement_type: str = "sampling"
    measurement_points: int = 1000
    # Initialization options, arguments of initialize()
    ordering: str = "A"


@dataclass
class Tracker:
    sweep: int = 0
    # error: List[float] = field(default_factory=lambda: [1.0])
    error: List[float] = []
    calls: List[int] = []
    iters: List[int] = []
    times: List[float] = []
    maxrank: List[int] = []
    effrank: List[float] = []

    def measure_time(self):
        pass


class Cross:
    def __init__(
        self,
        func: Callable,
        mesh: Mesh,
        mps: Optional[MPS] = None,
        cross_options: CrossOptions = CrossOptions(),
        tracker: Tracker = Tracker(),
    ):
        self.func = func
        self.mesh = mesh
        if mps is None:
            mps = random_mps([2] * sum(self.qubits), 1)
        self.mps = mps
        self.cross_options = cross_options
        self.tracker = tracker
        self.qubits = [int(np.log2(s)) for s in self.mesh.shape()[:-1]]
        self.structure = "binary" if np.all(self.mps.bond_dimensions() == 2) else "tt"

    def evaluate(self, indices: np.ndarray) -> np.ndarray:
        if self.structure == "binary":
            indices = binary_to_decimal(
                indices, self.qubits, self.cross_options.ordering
            )
        return np.array([self.func(self.mesh[idx]) for idx in indices])

    def sample(
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
        fiber = self.evaluate(indices).reshape((r1, s, r2), order="F")
        self.tracker.calls[self.tracker.sweep] += len(indices)
        return fiber

    def initialize(self):
        d = len(self.mps)
        self.I_physical = [
            np.arange(k, dtype=int).reshape(-1, 1)
            for k in self.mps.physical_dimensions()
        ]
        self.I_forward = [None for _ in range(len(self.mps) + 1)]
        self.I_backward = [None for _ in range(len(self.mps) + 1)]
        options = replace(self.cross_options, maxvol_minrank=0, maxvol_maxrank=0)

        # Forward pass
        R = np.ones((1, 1))
        for j in range(d):
            fiber = np.tensordot(R, self.mps[j], 1)
            self.mps[j], self.I_forward[j + 1], R = skeleton(
                fiber, self.I_physical[j], self.I_forward[j], options, ltr=True
            )
        self.mps[d - 1] = np.tensordot(self.mps[d - 1], R, 1)

        # Backward pass
        R = np.ones((1, 1))
        for j in range(d - 1, -1, -1):
            fiber = np.tensordot(self.mps[j], R, 1)
            self.mps[j], self.I_backward[j], R = skeleton(
                fiber, self.I_physical[j], self.I_backward[j + 1], options, ltr=True
            )
        self.mps[0] = np.tensordot(R, self.mps[0], 1)

    def sweep(self):
        d = len(self.mps)

        # Forward pass
        R = np.ones((1, 1))
        for j in range(d):
            fiber = self.sample(
                self.I_physical[j], self.I_forward[j], self.I_backward[j + 1]
            )
            self.mps[j], self.I_forward[j + 1], R = skeleton(
                fiber,
                self.I_physical[j],
                self.I_forward[j],
                self.cross_options,
                ltr=True,
            )
        self.mps[d - 1] = np.tensordot(self.mps[d - 1], R, 1)

        # Backward pass
        R = np.ones((1, 1))
        for j in range(d - 1, -1, -1):
            fiber = self.sample(
                self.I_physical[j], self.I_forward[j], self.I_backward[j + 1]
            )
            self.mps[j], self.I_backward[j], R = skeleton(
                fiber,
                self.I_physical[j],
                self.I_backward[j + 1],
                self.cross_options,
                ltr=False,
            )
        self.mps[0] = np.tensordot(R, self.mps[0], 1)

        self.tracker.sweep += 1
        self.tracker.maxrank.append(max(self.mps.bond_dimensions()))
        # self.tracker.effrank.append(effrank(self.mps.bond_dimensions()))
        # self.tracker.times.append(time)


def measure_sampling(
    cross: Cross, cross_options: CrossOptions, tracker: Tracker
) -> float:
    if tracker.iters[-1] == 0:
        tracker.sampling_indices = np.vstack(
            [
                np.random.choice(k, cross_options.measurement_points)
                for k in cross.mps.physical_dimensions()
            ]
        ).T
        tracker.samples = evaluate(cross, tracker.sampling_indices)

    Q = cross.mps[0][0, tracker.sampling_indices[:, 0], :]
    for i in range(1, len(cross.mps)):
        Q = np.einsum(
            "kq,qkr->kr", Q, cross.mps[i][:, tracker.sampling_indices[:, i], :]
        )
    error = np.linalg.norm(Q[:, 0] - tracker.samples) / np.linalg.norm(tracker.samples)

    tracker.error.append(error)
    return error


def measure(cross: Cross, cross_options: CrossOptions, tracker: Tracker) -> None:
    if cross_options.measurement_type == "sampling":
        error = measure_sampling(cross, cross_options, tracker)
        log_name = "Sampling error"
    # elif cross_options.measurement_type == "norm":
    #     error = measure_norm(cross, cross_options, tracker)  # TODO: Implement
    #     log_name = "Norm error"
    # elif cross_options.measurement_type == "integral":
    #     error = measure_integral(cross, cross_options, tracker)  # TODO: Implement
    #     log_name = "Integral error"
    else:
        raise ValueError("Invalid measurement type")

    # TODO: Implement as a log using the seemps debug parameter
    if cross_options.verbose:
        print(
            f"Sweep {tracker.sweep:<3} | "
            + f"Max χ {tracker.maxrank[-1]:>3} | "
            + f"{log_name} {error:.2E} | "
            + f"Function calls {tracker.calls[-1]:>8}"
        )


def is_converged(cross_options: CrossOptions, tracker: Tracker) -> bool:
    return (
        tracker.error[-1] < cross_options.tol
        or tracker.calls[-1] >= cross_options.maxcall
        or tracker.iters[-1] >= cross_options.maxiter
        or tracker.times[-1] >= cross_options.maxtime
        or tracker.maxrank[-1] >= cross_options.maxrank
    )


def do_cross_interpolation(cross: Cross):
    cross.initialize()
    while not cross.is_converged():
        cross.sweep()
        cross.measure()
    return cross


def do_cross_ideal(func: Callable, mesh: Mesh, mps0: Optional[MPS] = None):
    cross = Cross(func, mesh)
    cross.initialize(mps0=mps0, ordering="A")
    # Inicializa los índices I_forward, I_backward y el mps
    # tanto aleatorio como asignado. Reinicia el tracker.
    while not cross.is_converged(tol=1e-10):
        # Compara los argumentos de la función con el tracker.
        cross.sweep(**kwargs)
        cross.measure(measurement_type="sampling")
        # Definir varias funciones, como measure sampling, integral o norm.
        # Actualizan el tracker añadiendo nuevos valores.


# TOOLS


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


def reorder_tensor(tensor, qubits):
    pass


def sampling_error(mps, sampling_indices, sampled_vector):
    pass


def norm_error(mps, norm_prev):
    pass


# TODO: Incorporate inside sweep
# def presweep(cross: Cross, cross_options: CrossOptions) -> None:


def sweep(cross: Cross, cross_options: CrossOptions, tracker: Tracker) -> None:
    d = len(cross.mps)

    # Forward pass
    R = np.ones((1, 1))
    for j in range(d):
        fiber = sample(j, cross, tracker)
        cross, R = skeleton(fiber, cross, cross_options, ltr=True)
    cross.mps[d - 1] = np.tensordot(cross.mps[d - 1], R, 1)

    # Backward pass
    R = np.ones((1, 1))
    for j in range(d - 1, -1, -1):
        fiber = sample(j, cross, tracker)
        cross, R = skeleton(fiber, cross, cross_options, ltr=False)
    cross.mps[0] = np.tensordot(R, cross.mps[0], 1)

    tracker.sweep += 1
    tracker.maxrank.append(max(cross.mps.bond_dimensions()))
    # tracker.effrank.append(effrank(cross.mps))
    # tracker.times.append(time)


# TODO: Replace in
def _ones(k, m=1):
    return np.ones((k, m), dtype=int)
