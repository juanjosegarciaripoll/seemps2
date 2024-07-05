import numpy as np
from abc import ABC, abstractmethod
from typing import Callable, Union

from ..mesh import Interval, Mesh, mps_to_mesh_matrix
from ..sampling import evaluate_mps
from ...state import MPS


class BlackBox(ABC):
    """
    Abstract base class representing generic black-box functions.
    A black-box function represents an implicit representation of a function
    that can be indexed with indices similarly as a multidimensional array.
    These objects are fundamental for the efficient implementation of TCI algorithms.
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
    Black-box representing a multivariate scalar function discretized on an `Interval` or
    `Mesh` object. Each function degree of freedom is quantized in a given `base` and assigned
    a collection of MPS tensors. If the function is multivariate, the tensors are arranged
    according to the `mps_order`.

    Parameters
    ----------
    func : Callable
        The multivariate scalar function to be represented as MPS.
    domain : Union[Interval, Mesh]
        The domain where the function is discretized.
    base : int, default=2
        The required base or physical dimension of the MPS.
    mps_order : str, default='A'
        The order of the qubits of the MPS, either 'serial' ('A') or 'interleaved ('B').

    Example
    -------
        .. code-block:: python

        # Load a bivariate Gaussian function using some TCI variant.

        # Define the tensorized function following the convention of having the dimension index first.
        func = lambda tensor: np.exp(-(tensor[0]**2 + tensor[1]**2))

        # Define the bivariate domain implictly using `Interval` and `Mesh`
        start, stop = -1, 1
        n_qubits = 10
        interval = RegularInterval(start, stop, 2**n_qubits)
        mesh = Mesh([interval, interval])

        # Define the black box.
        black_box = BlackBoxLoadMPS(func, mesh)

        # Load the function in the given domain using some TCI variant (e.g. DMRG, Maxvol or Greedy).
        cross_results = cross_X(black_box)
        mps = cross_results.mps
    """

    def __init__(
        self,
        func: Callable,
        domain: Union[Interval, Mesh],
        base: int = 2,
        mps_order: str = "A",
    ):
        super().__init__(func)
        self.mesh = Mesh([domain]) if not isinstance(domain, Mesh) else domain
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
        # Transpose because of opposite conventions for mesh (dimension index last)
        # and cross (dimension index first).
        return self.func(self.mesh[mps_indices @ self.map_matrix].T)  # type: ignore


class BlackBoxLoadTT(BlackBox):
    """
    Black-box representing a multivariate scalar function discretized on a `Mesh` object
    following the tensor-train structure. Each function degree of freedom is assigned to
    one of each TT tensors.

    Parameters
    ----------
    func : Callable
        The multivariate scalar function to be represented as MPS.
    mesh : Mesh
        The domain where the function is discretized.

    Example
    -------
        .. code-block:: python

        # Load a bivariate Gaussian function using some TCI variant.

        # Define the tensorized function following the convention of having the dimension index first.
        func = lambda tensor: np.exp(-(tensor[0]**2 + tensor[1]**2))

        # Define the bivariate domain implictly using `Interval` and `Mesh`
        start, stop = -1, 1
        nodes = 1000
        interval = RegularInterval(start, stop, nodes)
        mesh = Mesh([interval, interval])

        # Define the black box.
        black_box = BlackBoxLoadTT(func, mesh)

        # Load the function in the given domain using some TCI variant (e.g. DMRG, Maxvol or Greedy).
        cross_results = cross_X(black_box)
        tensor_train = cross_results.mps
    """

    def __init__(
        self,
        func: Callable,
        mesh: Mesh,
    ):
        super().__init__(func)
        self.mesh = mesh
        self.base = np.inf  # type: ignore
        self.sites_per_dimension = [1 for _ in self.mesh.dimensions]
        self.sites = sum(self.sites_per_dimension)
        self.dimension = len(self.sites_per_dimension)
        self.physical_dimensions = [interval.size for interval in self.mesh.intervals]

    def __getitem__(self, mps_indices: np.ndarray) -> np.ndarray:
        self.evals += len(mps_indices)
        return self.func(self.mesh[mps_indices].T)  # type: ignore


class BlackBoxLoadMPO(BlackBox):
    """
    Black-box representing a 2-dimensional function discretized on a 2D `Mesh`
    and quantized in a MPO with physical dimensions given by `base_mpo`. Can be
    used to load operators in MPO using tensor cross-interpolation. In practice,
    this object is equivalently represented as a MPS with physical dimensions
    of size `base_mpo**2`, whose indices can be subsequently split to form
    the required MPO.

    Parameters
    ----------
    func : Callable
        The bivariate scalar function to be represented as MPO.
    mesh : Mesh
        The two-dimensional discretization where the function is discretized.
    base_mpo : int, default=2
        The required physical dimension of each index of the MPO.
    is_diagonal : bool, default=True
        Flag that helps in the convergence of TCI for diagonal operators by restricting
        the convergence evaluation to the main diagonal.

    Example
    -------
        .. code-block:: python

        # Load a 2D Gaussian function in a non-diagonal MPO using some TCI variant.

        # Define the tensorized function following the convention of having the dimension index first.
        func = lambda tensor: np.exp(-(tensor[0]**2 + tensor[1]**2))

        # Define the bivariate domain implictly using `Interval` and `Mesh`.
        start, stop = -1, 1
        num_qubits = 10
        interval = RegularInterval(start, stop, 2**n)
        mesh = Mesh([interval, interval])

        # Define the black box.
        black_box = BlackBoxLoadMPO(func, mesh)

        # Load the function in the given domain using some tci variant (e.g. DMRG, Maxvol or Greedy).
        cross_results = cross_X(black_box)
        mpo = mps_as_mpo(cross_results.mps) # Unfold into a MPO.
    """

    # TODO: Generalize for multivariate MPOs.
    def __init__(
        self,
        func: Callable,
        mesh: Mesh,
        base_mpo: int = 2,
        is_diagonal: bool = False,
    ):
        super().__init__(func)
        self.mesh = mesh
        self.base_mpo = base_mpo
        self.is_diagonal = is_diagonal

        if not (mesh.dimension == 2 and mesh.dimensions[0] == mesh.dimensions[1]):
            raise ValueError("The mesh must be bivariate for a 1d MPO")
        self.sites = int(np.emath.logn(self.base_mpo, mesh.dimensions[0]))
        if not self.base_mpo**self.sites == mesh.dimensions[0]:
            raise ValueError(f"The mesh cannot be quantized with base {self.base_mpo}")

        self.base = base_mpo**2
        self.dimension = 1
        self.physical_dimensions = [self.base] * self.sites
        self.sites_per_dimension = [self.sites]
        self.map_matrix = mps_to_mesh_matrix(
            self.sites_per_dimension, base=self.base_mpo
        )

        # If the MPO is diagonal, restrict the allowed indices for random sampling to the main diagonal.
        self.allowed_indices = (
            [s * base_mpo + s for s in range(base_mpo)] if self.is_diagonal else None
        )

    def __getitem__(self, mps_indices: np.ndarray) -> np.ndarray:
        self.evals += len(mps_indices)
        row_indices = (mps_indices // self.base_mpo) @ self.map_matrix
        col_indices = (mps_indices % self.base_mpo) @ self.map_matrix
        mesh_indices = np.hstack((row_indices, col_indices))
        return self.func(*self.mesh[mesh_indices].T)  # type: ignore


class BlackBoxComposeMPS(BlackBox):
    """
    Black-box representing the composition of a multivariate scalar function with
    a collection of MPS objects.

    Parameters
    ----------
    func : Callable
        The function to compose with the collection of MPS objects. Must be
        scalar, and each of its degrees of freedom must refer to each of the MPS
        in the `mps_list` collection. For example, `f(x, y) = sin(x + y**2)`
        acts on two MPS representing respectively `x` and `y`.
    mps_list : list[MPS]
        A list of MPS of the same physical dimension, to be composed with `func`.
        Their physical dimensions are assumed similar and constant. The number of MPS
        must match the dimension of `func`.

    Example
    -------
    .. code-block:: python

        # Use TCI to compose a three-dimensional function with three MPS.

        # Assume the three initial MPS are given and are of the same structure.
        mps_0, mps_1, mps_2 = ...

        # Define the three dimensional function by its action on the MPS.
        func = lambda v: v[0]**2 + np.sin(v[0]*v[1]) + np.cos(v[0]*v[2])

        # Define the black-box.
        black_box = BlackBoxComposeMPS(func, [mps_0, mps_1, mps_2])

        # Compose the three MPS with the function `func`.
        cross_results = cross_X(black_box)
        mps = cross_results.mps
    """

    def __init__(self, func: Callable, mps_list: list[MPS]):
        super().__init__(func)

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
