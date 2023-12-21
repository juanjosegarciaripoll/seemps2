from abc import ABC, abstractmethod
from itertools import product
from typing import Callable, List, Tuple

import numpy as np


class Interval(ABC):
    """Interval Abstract Base Class.

    This abstracts an Interval object, which represents implicitly an
    interval discretized along N points within two endpoints start and stop.
    """

    def __init__(self, start: float, stop: float, size: int):
        self.start = start
        self.stop = stop
        self.size = size

    @abstractmethod
    def __getitem__(self, idx: int) -> float:
        ...

    def to_vector(self) -> np.ndarray:
        return np.array([self[idx] for idx in range(self.size)])


class RegularClosedInterval(Interval):
    """Equispaced discretization between [start, stop]."""

    def __init__(self, start: float, stop: float, size: int):
        super().__init__(start, stop, size)
        self.step = (stop - start) / (size - 1)

    def __getitem__(self, idx: int) -> float:
        if not (0 <= idx < self.size):
            raise IndexError("Index out of range")
        return idx * self.step + self.start


class RegularHalfOpenInterval(Interval):
    """Equispaced discretization between [start, stop)."""

    def __init__(self, start: float, stop: float, size: int):
        super().__init__(start, stop, size)
        self.step = (stop - start) / size

    def __getitem__(self, idx: int) -> float:
        if not (0 <= idx < self.size):
            raise IndexError("Index out of range")
        return idx * self.step + self.start


class ChebyshevZerosInterval(Interval):
    """Irregular discretization given by an affine map between the
    zeros of the N-th Chebyshev polynomial in [-1, 1] to (start, stop)."""

    def __init__(self, start: float, stop: float, size: int):
        super().__init__(start, stop, size)

    def __getitem__(self, idx: int) -> float:
        if not (0 <= idx < self.size):
            raise IndexError("Index out of range")
        zero = np.cos(np.pi * (2 * (self.size - idx) - 1) / (2 * self.size))
        return (self.stop - self.start) * (zero + 1) / 2 + self.start


class Mesh:
    """Multidimensional mesh object.

    This represents a multidimensional mesh which can be understood as the
    implicit tensor given by the cartesian product of a collection of intervals.

    Parameters
    ----------
    intervals : List[Interval]
        A list of Interval objects representing the discretizations along each dimension.
    """

    def __init__(self, intervals: List[Interval]):
        self.intervals = intervals
        self.dimension = len(intervals)

    def __getitem__(self, indices: Tuple[int, ...]):
        if len(indices) != self.dimension:
            raise ValueError("Incorrect index size")
        return np.array(
            [interval[idx] for interval, idx in zip(self.intervals, indices)]
        )

    def shape(self):
        return tuple(interval.size for interval in self.intervals) + (self.dimension,)

    def to_tensor(self):
        return np.array(list(product(*self.intervals))).reshape(self.shape())


def mps_indices_to_mesh_indices(
    mesh: Mesh, mps_indices: np.ndarray, mps_ordering: str = "A"
) -> np.ndarray:
    """
    Converts indices from MPS format to Mesh (decimal) format.
    If the mesh represents a multivariate domain, the indices
    depend on the ordering of the MPS.

    Parameters
    ----------
    mesh : Mesh
        The mesh object defining the spatial domain.
    mps_indices : np.ndarray
        Array of indices in MPS format.
    mps_ordering : str, default 'A'
        The ordering of the MPS, either 'A' or 'B'.

    Returns
    -------
    mesh_indices : np.ndarray
        Array of indices in the mesh corresponding to the MPS indices.
    """

    bits_to_int = lambda bits: int("".join(map(str, bits)), 2)
    sites_per_dimension = [int(np.log2(size)) for size in mesh.shape()[:-1]]
    dims = len(sites_per_dimension)
    mesh_indices = []
    for dim, sites in enumerate(sites_per_dimension):
        if mps_ordering == "A":
            slice = np.arange(dim * sites, (dim + 1) * sites)
        elif mps_ordering == "B":
            slice = np.arange(dim, dims * sites, dims)
        else:
            raise ValueError("Invalid mps_ordering")
        mesh_indices.append(
            np.array([bits_to_int(bits) for bits in mps_indices[:, slice]])
        )
    return np.column_stack(mesh_indices)


# TODO: Think if this should be included in the library
def reorder_tensor(tensor: np.ndarray, sites_per_dimension: List[int]) -> np.ndarray:
    """
    Reorders a given tensor between the MPS orderings 'A' and 'B' by transposing its axes.

    This reshapes the input tensor into a MPS format, transposes its axes according to the
    MPS ordering and specified sites per dimension, and reshapes it back to the original tensor shape.

    Parameters
    ----------
    tensor : np.ndarray
        The tensor to be reordered.
    sites_per_dimension : List[int]
        A list specifying the number of sites for each dimension of the tensor.

    Returns
    -------
    np.ndarray
        The tensor reordered from order 'A' to 'B' or vice versa.
    """
    dimensions = len(sites_per_dimension)
    shape_orig = tensor.shape
    tensor = tensor.reshape([2] * sum(sites_per_dimension))
    axes = [
        np.arange(idx, dimensions * n, dimensions)
        for idx, n in enumerate(sites_per_dimension)
    ]
    axes = [item for items in axes for item in items]
    tensor = np.transpose(tensor, axes=axes)
    return tensor.reshape(shape_orig)


def sample_mesh(
    func: Callable, mesh: Mesh, mps_indices: np.ndarray, mps_ordering: str
) -> np.ndarray:
    mesh_indices = mps_indices_to_mesh_indices(mesh, mps_indices, mps_ordering)
    return np.array([func(mesh[idx]) for idx in mesh_indices]).flatten()
