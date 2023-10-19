from dataclasses import dataclass, field
import numpy as np
from typing import Callable, List, Optional
from .maxvol import maxvol
from .mesh import Mesh
from ..state import MPS, random_mps


@dataclass
class CrossOptions:
    tol: float = 1e-10
    maxcall: int = 10000000
    maxiter: int = 1000
    maxrank: int = 1000
    maxvol_minrank: int = 0
    maxvol_maxrank: int = 1
    maxvol_sqr_tau: float = 1.05
    maxvol_sqr_maxiter: int = 100
    maxvol_rect_tau: float = 1.1
    maxvol_rect_maxiter: int = 100
    measurement_type: str = "sampling"
    measurement_points: int = 1000


@dataclass
class Cross:
    func: Callable
    mesh: Mesh
    mps0: Optional[MPS] = None

    def __post_init__(self):
        if self.mps0 is None:
            qubits = [int(np.log2(s)) for s in self.mesh.shape()[:-1]]
            self.mps0 = random_mps([2] * sum(qubits), 1)


@dataclass
class Tracker:
    mps0: MPS
    error: List[float] = field(default_factory=lambda: [1.0])
    calls: List[int] = field(default_factory=lambda: [0])
    iters: List[int] = field(default_factory=lambda: [0])

    def __post_init__(self):
        self.maxrank: List[int] = [max(self.mps0.bond_dimensions())]
        # self.effrank: List[float] = [effective_rank(mps0)] TODO: Implement


# TODO: Merge with evaluate
def sample(cross: Cross) -> np.ndarray:
    return sampled_tensor


# TODO: Merge with sample and rename to sample
def evaluate(cross: Cross, tracker: Tracker) -> np.ndarray:
    return evaluated_tensor


def sweep(cross: Cross, cross_options: CrossOptions) -> None:
    d = len(cross.mps0)

    # Forward pass
    R = np.ones((1, 1))
    for j in range(d):
        fiber = evaluate(j, cross, tracker, ltr=True)


def measure(cross: Cross, cross_options: CrossOptions, tracker: Tracker) -> None:
    if cross_options.measurement_type == "sampling":
        if tracker.iters[-1] == 0:
            tracker.sampling_indices = np.vstack(
                [
                    np.random.choice(k, cross_options.measurement_points)
                    for k in cross.mps0.physical_dimensions()
                ]
            ).T
            tracker.sampled_vector = sample(tracker.sampling_indices)
        error = sampling_error(
            cross.mps0, tracker.sampling_indices, tracker.sampled_vector
        )
        tracker.error.append(error)
        log_name = "Sampling error"
    elif cross_options.measurement_type == "norm":  # TODO: Implement
        pass
    else:
        raise ValueError("Invalid measurement type")

    if (
        cross_options.verbose
    ):  # TODO: Implement as a log using the seemps debug parameter
        print(
            f"Sweep {tracker.sweeps[-1]:<3} | "
            + f"Max χ {tracker.maxrank[-1]:>3} | "
            + f"{log_name} {error:.2E} | "
            + f"Function calls {tracker.calls[-1]:>8}"
        )


def is_converged(cross_options: CrossOptions, tracker: Tracker) -> bool:
    return (
        tracker.error[-1] < cross_options.tol
        or tracker.calls[-1] >= cross_options.maxcall
        or tracker.iters[-1] >= cross_options.maxiter
        or tracker.maxrank[-1] >= cross_options.maxrank
    )


def run_cross(cross: Cross, cross_options: CrossOptions = CrossOptions()):
    tracker = Tracker(cross.mps0)
    while not is_converged(cross_options, tracker):
        sweep(cross, cross_options)
        measure(cross, tracker)
    return cross, tracker


# TOOLS


def indices_bin_to_dec(indices, qubits, ordering):
    pass


def reorder_tensor(tensor, qubits):
    pass


def sampling_error(mps, sampling_indices, sampled_vector):
    pass


def norm_error(mps, norm_prev):
    pass


# TODO: Incorporate inside sweep
# def presweep(cross: Cross, cross_options: CrossOptions) -> None:
