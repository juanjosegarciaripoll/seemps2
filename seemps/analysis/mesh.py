from abc import ABC, abstractmethod
from itertools import product
from typing import List, Union, Tuple

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
        self.transformation_matrix = None

    def __getitem__(self, indices: Union[Tuple[int, ...], np.ndarray]):
        if isinstance(indices, np.ndarray):
            indices = np.atleast_2d(indices)
            if indices.shape[1] != self.dimension:
                raise ValueError("Incorrect index shape for NumPy array")
            result = np.empty_like(indices, dtype=float)
            for i, interval in enumerate(self.intervals):
                result[:, i] = indices[:, i] * interval.step + interval.start
            return result[0] if indices.ndim == 1 else result

        elif isinstance(indices, tuple):
            if len(indices) != self.dimension:
                raise ValueError("Incorrect number of indices")
            return np.array(
                [interval[idx] for interval, idx in zip(self.intervals, indices)]
            )
        else:
            raise TypeError("Indices must be a tuple or NumPy array")

    def binary_transformation_matrix(self, order="A", base=2) -> np.ndarray:
        """
        Constructs and returns a binary transformation matrix based on the specified order and base.
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
                T = []
                for i in range(max(sites)):
                    diagonal = [2 ** (n - i - 1) if n > i else 0 for n in sites]
                    T.append(np.diag(diagonal))
                T = np.vstack(T)
                T = T[~np.all(T <= 0, axis=1)]
                self.transformation_matrix = T
        return self.transformation_matrix

    def shape(self):
        return tuple(interval.size for interval in self.intervals) + (self.dimension,)

    def to_tensor(self):
        return np.array(list(product(*self.intervals))).reshape(self.shape())
