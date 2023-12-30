from __future__ import annotations
from abc import ABC, abstractmethod
from itertools import product
from typing import Union, Optional, Sequence
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
    """

    intervals: list[Interval]
    dimension: int
    transformation_matrix: Optional[np.ndarray]

    def __init__(self, intervals: list[Interval]):
        self.intervals = intervals
        self.dimension = len(intervals)
        self.transformation_matrix = None

    def __getitem__(
        self, indices: Union[Sequence[int], np.ndarray]
    ) -> Union[float, Vector]:
        """Return the vector of coordinates of a point in the mesh.

        Parameters
        ----------
        indices : Sequence[int] | np.ndarray[dim=1]
            A sequence of integers indexing a point in the mesh.

        Returns
        -------
        point : float | np.ndarray[float]
            A vector of coordinates into the mesh, or a scalar if this
            a 1D field.
        """
        if (isinstance(indices, np.ndarray) and indices.ndim != 1) or len(
            indices
        ) != self.dimension:
            raise ValueError("Invalid index into the mesh")
        if self.dimension == 1:
            return self.intervals[0][indices[0]]
        else:
            return np.asarray(
                [interval[i] for i, interval in zip(indices, self.intervals)]
            )

    def binary_transformation_matrix(self, order="A", base=2) -> np.ndarray:
        """
        Constructs and returns a binary transformation matrix based on the
        specified order and base.
        """
        sites = [int(np.emath.logn(base, s)) for s in self.shape()[:-1]]
        if self.transformation_matrix is None:
            if order == "A":
                T = np.zeros((sum(sites), len(sites)), dtype=int)
                start = 0
                for m, n in enumerate(sites):
                    T[start : start + n, m] = 2 ** np.arange(n)[::-1]
                    start += n
                self.transformation_matrix = T
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
                self.transformation_matrix = T
            else:
                raise ValueError("Invalid MPS order")
            return T
        return self.transformation_matrix

    def shape(self):
        return tuple(interval.size for interval in self.intervals) + (self.dimension,)

    def to_tensor(self):
        return np.array(list(product(*self.intervals))).reshape(self.shape())
