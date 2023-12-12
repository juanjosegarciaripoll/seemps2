from abc import ABC, abstractmethod
from typing import List, Tuple
import numpy as np
from itertools import product


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
        if not (0 <= idx < self.size):
            raise IndexError("Index out of range")

    def to_vector(self) -> np.ndarray:
        return np.array([self[idx] for idx in range(self.size)])


class RegularClosedInterval(Interval):
    """Equispaced discretization between [start, stop]."""

    def __init__(self, start: float, stop: float, size: int):
        super().__init__(start, stop, size)
        self.type = "closed"
        self.step = (stop - start) / (size - 1)

    def __getitem__(self, idx: int) -> float:
        super().__getitem__(idx)
        return idx * self.step + self.start


class RegularHalfOpenInterval(Interval):
    """Equispaced discretization between [start, stop)."""

    def __init__(self, start: float, stop: float, size: int):
        super().__init__(start, stop, size)
        self.type = "open"
        self.step = (stop - start) / size

    def __getitem__(self, idx: int) -> float:
        super().__getitem__(idx)
        return idx * self.step + self.start


class ChebyshevZerosInterval(Interval):
    """Irregular discretization given by an affine map between the
    zeros of the N-th Chebyshev polynomial in [-1, 1] to (start, stop)."""

    def __init__(self, start: float, stop: float, size: int):
        self.type = "zeros"
        super().__init__(start, stop, size)

    def __getitem__(self, idx: int) -> float:
        super().__getitem__(idx)
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
