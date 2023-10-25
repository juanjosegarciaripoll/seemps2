from dataclasses import dataclass, replace
import numpy as np
from copy import deepcopy
from typing import Callable, List, Optional, Tuple
from .maxvol import maxvol_sqr, maxvol_rct
from .mesh import Mesh
from ..state import MPS, random_mps


@dataclass
class CrossStrategy:
    tol: float = 1e-10
    maxiter: int = 100
    maxrank: int = 100
    mps_ordering: str = "A"
    maxvol_sqr_tau: float = 1.05
    maxvol_sqr_maxiter: int = 100
    maxvol_rct_tau: float = 1.10
    maxvol_rct_minrank: int = 1
    maxvol_rct_maxrank: int = 1


@dataclass
class Cross:
    func: Callable
    mesh: Mesh
    mps: MPS

    def __post_init__(self):
        shape_mps = self.mps.physical_dimensions()
        shape_mesh = self.mesh.shape()[:-1]
        if np.prod(shape_mps) == np.prod(shape_mesh) and all(
            dim == 2 for dim in shape_mps
        ):
            self.structure = "binary"
        elif shape_mps == shape_mesh:
            self.structure = "tt"
        else:
            raise ValueError("Non-matching mesh and initial MPS")
        self.sites = len(self.mps)


def _initialize(cross: Cross, cross_strategy: CrossStrategy) -> None:
    pass


def _sweep(cross: Cross, cross_strategy: CrossStrategy) -> None:
    pass


def _skeleton(
    cross: Cross,
    j: int,
    cross_strategy: CrossStrategy,
    forward: bool,
    evaluate: bool = True,
) -> np.ndarray:
    pass


def _sample(cross: Cross, j: int, cross_strategy: CrossStrategy) -> np.ndarray:
    pass


def _evaluate(cross: Cross, indices: np.ndarray) -> np.ndarray:
    pass


def _binary2decimal(
    indices: np.ndarray, qubits: List[int], mps_ordering: str
) -> np.ndarray:
    pass


def _error_sampling(cross: Cross, sampling_points: int = 1000) -> float:
    pass


def _error_norm2(cross: Cross) -> float:
    pass


def _error_integral(cross: Cross) -> float:
    pass


def _converged(cross: Cross, cross_strategy: CrossStrategy) -> bool:
    pass


def reorder_tensor(tensor: np.ndarray, qubits: List[int]) -> np.ndarray:
    pass


def cross_interpolation(
    func: Callable,
    mesh: Mesh,
    mps: Optional[MPS] = None,
    cross_strategy: CrossStrategy = CrossStrategy(),
) -> MPS:
    if mps is None:
        mesh_size = mesh.shape()[:-1]
        _is_power_of_two = lambda s: (s != 0) and (s & (s - 1) == 0)
        if not all(_is_power_of_two(s) for s in mesh_size):
            raise ValueError("The mesh size must be a power of two")
        sites = sum([int(np.log2(s)) for s in mesh_size])
        mps = random_mps([2] * sites, 1, rng=np.random.default_rng(42))
    cross = Cross(func, mesh, mps)
    cross = _initialize(cross, cross_strategy)
    while not _converged(cross, cross_strategy):
        _sweep(cross, cross_strategy)
        cross.error = _error_norm2(cross)
    return cross.mps
