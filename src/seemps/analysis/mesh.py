from __future__ import annotations
from abc import ABC, abstractmethod
from itertools import product
from typing import Union, Sequence, Iterator
from ..typing import Vector

import numpy as np


class Interval(ABC):
    """Interval Abstract Base Class.

    This abstracts an Interval object, which represents implicitly an interval
    discretized along N points within two endpoints start and stop. Intervals
    act like sequences of numbers denoting points along the interval. They
    can be accessed as in `i[0]`, `i[1]`,... up to `i[size-1]` and they can
    be converted to other sequences, as in `list(i)`, or iterated over.
    """

    start: float
    stop: float
    size: int

    def __init__(self, start: float, stop: float, size: int):
        self.start = start
        self.stop = stop
        self.size = size

    def __len__(self) -> int:
        return self.size

    @abstractmethod
    def __getitem__(self, idx: int) -> float:
        ...

    def to_vector(self) -> np.ndarray:
        return np.array([self[idx] for idx in range(self.size)])

    def map_to(self, start: float, stop: float) -> Interval:
        return type(self)(start, stop, self.size)

    def update_size(self, size: int) -> Interval:
        return type(self)(self.start, self.stop, size)

    def __iter__(self) -> Iterator:
        return (self[i] for i in range(self.size))


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
    intervals : list[Interval]
        A list of Interval objects representing the discretizations along each
        dimension.

    Attributes
    ----------
    intervals : list[Interval]
        The supplied list of intervals.
    dimension : int
        Dimension of the space in which this mesh is embedded.
    shape : tuple[int]
        Shape of the equivalent tensor this Mesh can be converted to.
    dimensions : tuple[int]
        Tuple of the sizes of each interval
    """

    intervals: list[Interval]
    dimension: int
    shape: tuple[int, ...]
    dimensions: tuple[int, ...]
    _v: list[np.ndarray]

    def __init__(self, intervals: list[Interval]):
        def unit_vector(n: int, L: int):
            v = np.zeros(L)
            v[n] = 1.0
            return v

        self.intervals = intervals
        self.dimension = len(intervals)
        self.dimensions = tuple(interval.size for interval in self.intervals)
        #
        # The field _v contains a list of arrays, each with size Ni x d
        # where Ni is the size of the i-th interval, and d is the dimension
        # of the mesh. The _v[i] array only has components in _v[i][:,i]
        # This way, we can construct points in the mesh going from integers
        # (i1, i2, ..., id) to vectors (x1, x2, ..., x2) as
        #   x = _v[0][i1,:] + _v[1][i2,:] + ... + _v[d-1][id,:]
        #
        self._v = [
            interval.to_vector().reshape(-1, 1) * unit_vector(i, self.dimension)
            for i, interval in enumerate(intervals)
        ]

    def __getitem__(
        self, indices: Union[Sequence[int], np.ndarray]
    ) -> Union[float, Vector]:
        """Return the vector of coordinates of a point in the mesh.

        The input can take different shapes for a D-dimensional mesh:
        * It can be a single integer, denoting a point in a 1D mesh.
        * It can be a vector of D coordinates, indexing a single point
          in the mesh.
        * It can be an N-dimensional array, denoting multiple points.
          The N-th dimension of the array must have size `D`, because
          it is the one indexing points in the Mesh.

        Parameters
        ----------
        indices : int | ArrayLike
            An integer, or an array-like structure indexing points
            in the mesh.

        Returns
        -------
        points : float | np.ndarray[float]
            Coordinates of one or more points.
        """
        if isinstance(indices, int):
            if self.dimension == 1:
                return self._v[0][indices]
            raise IndexError("Invalid index into a Mesh")
        indices = np.asarray(indices)
        if indices.shape[-1] != self.dimension:
            raise IndexError("Invalid index into a Mesh")
        else:
            return sum(self._v[n][indices[..., n]] for n in range(self.dimension))

    def mps_to_mesh_matrix(self, order="A", base=2) -> np.ndarray:
        """
        Returns a matrix that transforms an array of MPS indices
        to an array of Mesh indices based on the specified order and base.
        """
        sites = [int(np.emath.logn(base, s)) for s in self.dimensions]
        if order == "A":
            T = np.zeros((sum(sites), len(sites)), dtype=int)
            start = 0
            for m, n in enumerate(sites):
                T[start : start + n, m] = 2 ** np.arange(n)[::-1]
                start += n
            return T
        elif order == "B":
            # Strategy: stack diagonal matrices and remove unwanted rows.
            # TODO: Improve this logic.
            T = np.vstack(
                [
                    np.diag([2 ** (n - i - 1) if n > i else 0 for n in sites])
                    for i in range(max(sites))
                ]
            )
            T = T[~np.all(T <= 0, axis=1)]
            return T
        else:
            raise ValueError("Invalid MPS order")

    def to_tensor(self):
        return np.array(list(product(*self.intervals))).reshape(
            *self.dimensions, self.dimension
        )
