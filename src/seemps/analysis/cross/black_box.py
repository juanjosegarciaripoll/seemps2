import numpy as np
from abc import ABC, abstractmethod
from typing import Callable, Union

from ..mesh import Interval, Mesh, mps_to_mesh_matrix
from ..sampling import evaluate_mps
from ...state import MPS
from ...mpo import MPO


class BlackBox(ABC):
    """
    Abstract base class representing generic black-box functions.
    These are generic objects that can be evaluated by indexing them with indices
    similarly as a multidimensional array or a Mesh object. They serve as arguments for the
    tensor cross-interpolation algorithms.
    """

    base: int
    sites: int
    dimension: int
    physical_dimensions: list
    sites_per_dimension: list

    def __init__(self, func: Callable):
        self.func = func
        self.evals = 0

    @abstractmethod
    def __getitem__(self, mps_indices: np.ndarray) -> np.ndarray: ...


class BlackBoxLoadMPS(BlackBox):
    """
    Black-box representing the quantization of a multivariate function discretized on a Mesh
    with a given base and mps_order. Used to load the black-box function in a MPS.
    """

    def __init__(
        self,
        func: Callable,
        domain: Union[Interval, Mesh],
        base: int = 2,
        mps_order: str = "A",
    ):
        super().__init__(func)
        self.mesh = Mesh([domain]) if isinstance(domain, Interval) else domain
        self.base = base
        self.mps_order = mps_order

        self.sites_per_dimension = [
            int(np.emath.logn(base, s)) for s in self.mesh.dimensions
        ]
        if not all(
            base**n == N for n, N in zip(self.sites_per_dimension, self.mesh.dimensions)
        ):
            raise ValueError(f"The mesh cannot be quantized with base {base}")
        self.sites = sum(self.sites_per_dimension)
        self.dimension = len(self.sites_per_dimension)
        self.physical_dimensions = [self.base] * self.sites
        self.map_matrix = mps_to_mesh_matrix(
            self.sites_per_dimension, self.mps_order, self.base
        )

    def __getitem__(self, mps_indices: np.ndarray) -> np.ndarray:
        self.evals += len(mps_indices)
        # TODO: The transpose is necessary here because the mesh convention (dimension index last)
        # and the cross convention (dimension index first) are opposite. This should be fixed.
        return self.func(self.mesh[mps_indices @ self.map_matrix].T)  # type: ignore


class BlackBoxLoadMPO(BlackBox):
    """
    Black-box representing the quantization of a multivariate function discretized on a Mesh
    with a given base and mps_order. Used to load the black-box function in a MPO.

    As opposed to BlackBoxMesh2MPS, this class represents an operator by assigning
    pairs of variables to the operator rows and columns. At the moment it only works
    for univariate MPOs, that is, for bivariate functions f(x, y) and bivariate meshes
    Mesh([interval_x, interval_y]) representing the individual elements of the operator.
    """

    # TODO: Generalize for multivariate MPOs.
    def __init__(
        self,
        func: Callable,
        mesh: Mesh,
        base_mpo: int = 2,
        mpo_order: str = "A",
        is_diagonal: bool = False,
    ):
        super().__init__(func)
        self.mesh = mesh
        self.base_mpo = base_mpo
        self.mpo_order = mpo_order
        self.is_diagonal = is_diagonal

        # Check if the mesh is bivariate (representing a 1d MPO)
        if not (mesh.dimension == 2 and mesh.dimensions[0] == mesh.dimensions[1]):
            raise ValueError("The mesh must be bivariate for a 1d MPO")

        # Check if the mesh can be quantized with the given base
        self.sites = int(np.emath.logn(self.base_mpo, mesh.dimensions[0]))
        if not self.base_mpo**self.sites == mesh.dimensions[0]:
            raise ValueError(f"The mesh cannot be quantized with base {self.base_mpo}")

        # Define the structure of the equivalent MPS
        self.base = base_mpo**2
        self.dimension = 1
        self.physical_dimensions = [self.base] * self.sites
        self.sites_per_dimension = [self.sites]

        # If the MPO is diagonal, restrict the randomly sampled indices for evaluating
        # the error to the diagonal (s_i = s_j) => s = i*s + i, i = 0, 1, ..., s-1
        self.allowed_sampling_indices = (
            [s * base_mpo + s for s in range(base_mpo)] if self.is_diagonal else None
        )

        # Compute the transformation matrix (for the MPO indices with base_mpo)
        self.map_matrix = mps_to_mesh_matrix(
            self.sites_per_dimension, base=self.base_mpo
        )

    def __getitem__(self, mps_indices: np.ndarray) -> np.ndarray:
        self.evals += len(mps_indices)
        row_indices = (mps_indices // self.base_mpo) @ self.map_matrix
        col_indices = (mps_indices % self.base_mpo) @ self.map_matrix
        mesh_indices = np.hstack((row_indices, col_indices))
        return self.func(*self.mesh[mesh_indices].T)  # type: ignore


class BlackBoxComposeMPS(BlackBox):
    """
    Black-box representing the composition of a scalar function on a collection of MPS.
    The function must act on the list of MPS and these must be of same physical dimensions.
    """

    def __init__(self, func: Callable, mps_list: list[MPS]):
        super().__init__(func)

        # Assert that the physical dimensions are the same for all MPS
        self.physical_dimensions = mps_list[0].physical_dimensions()
        for mps in mps_list:
            if mps.physical_dimensions() != self.physical_dimensions:
                raise ValueError("All MPS must have the same physical dimensions.")

        self.base = self.physical_dimensions[0]  # Assume constant
        self.mps_list = mps_list
        self.sites = len(self.physical_dimensions)
        self.dimension = 1
        self.sites_per_dimension = [self.sites]

    def __getitem__(self, mps_indices: np.ndarray) -> np.ndarray:
        self.evals += len(mps_indices)
        mps_values = []
        for mps in self.mps_list:
            mps_values.append(evaluate_mps(mps, mps_indices))
        return self.func(mps_values)


class BlackBoxComposeMPO(BlackBox):
    """
    Black-box representing the composition of a scalar function on a collection of MPO.
    This is actually a good application of MPO Chebyshev approximation.

    Note: The function of a matrix is not equivalent to the function of its elements, so this cannot be
    performed in a straightforward manner similarly as BlackBoxMPS.
    Possible alternatives are methods such as:
    - Lagrange-Sylvester interpolation (requires eigenvalues).
    - Cauchy contour integral formula.
    etc.
    """

    def __init__(self, func: Callable, mpo_list: MPO):
        raise NotImplementedError
