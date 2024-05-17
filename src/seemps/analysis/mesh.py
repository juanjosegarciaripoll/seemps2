from __future__ import annotations
from abc import ABC, abstractmethod
from itertools import product
from typing import Union, Sequence, Iterator, overload
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

    def _validate_index(self, idx):
        if isinstance(idx, int):
            if not (0 <= idx < self.size):
                raise IndexError("Index out of range")
        elif isinstance(idx, np.ndarray):
            if not np.all((0 <= idx) & (idx < self.size)):
                raise IndexError("Index out of range")
        else:
            raise TypeError("Index must be an integer or a NumPy array")

    @overload
    def __getitem__(self, idx: np.ndarray) -> np.ndarray: ...

    @overload
    def __getitem__(self, idx: int) -> float: ...

    @abstractmethod
    def __getitem__(self, idx: Union[int, np.ndarray]) -> Union[float, np.ndarray]: ...  # type: ignore

    def to_vector(self) -> np.ndarray:
        return np.array([self[idx] for idx in range(self.size)])

    def map_to(self, start: float, stop: float) -> Interval:
        return type(self)(start, stop, self.size)

    def update_size(self, size: int) -> Interval:
        return type(self)(self.start, self.stop, size)

    def __iter__(self) -> Iterator:
        return (self[i] for i in range(self.size))


class RegularInterval(Interval):
    """Equispaced discretization between start and stop.
    The left and right boundary conditions can be set open or closed.
    Defaults to a closed-left, open-right interval [start, stop).
    """

    def __init__(
        self,
        start: float,
        stop: float,
        size: int,
        endpoint_right: bool = False,
        endpoint_left: bool = True,
    ):
        super().__init__(start, stop, size)
        self.endpoint_left = endpoint_left
        self.endpoint_right = endpoint_right
        if endpoint_left and endpoint_right:
            self.num_steps = self.size - 1
        elif endpoint_left or endpoint_right:
            self.num_steps = self.size
        else:
            self.num_steps = self.size + 1
        self.step = (stop - start) / self.num_steps
        self.start_displaced = (
            self.start if self.endpoint_left else self.start + self.step
        )

    @overload
    def __getitem__(self, idx: np.ndarray) -> np.ndarray: ...

    @overload
    def __getitem__(self, idx: int) -> float: ...

    def __getitem__(self, idx: Union[int, np.ndarray]) -> Union[float, np.ndarray]:
        super()._validate_index(idx)
        return self.start_displaced + idx * self.step


class ChebyshevInterval(Interval):
    """Irregular discretization given by an affine map between the
    nodes (zeros or extrema) of the N-th Chebyshev polynomial in [-1, 1] to (start, stop).
    If `endpoints=True` returns the Chebyshev extrema defined in the closed interval [a, b].
    Else, returns the Chebyshev zeros defined in the open interval (a, b)."""

    def __init__(self, start: float, stop: float, size: int, endpoints: bool = False):
        super().__init__(start, stop, size)
        self.endpoints = endpoints

    @overload
    def __getitem__(self, idx: np.ndarray) -> np.ndarray: ...

    @overload
    def __getitem__(self, idx: int) -> float: ...

    def __getitem__(self, idx: Union[int, np.ndarray]) -> Union[float, np.ndarray]:
        super()._validate_index(idx)
        if self.endpoints:  # Chebyshev extrema
            nodes = np.cos(np.pi * idx / (self.size - 1))
        else:  # Chebyshev zeros
            nodes = np.cos(np.pi * (2 * idx + 1) / (2 * self.size))
        return array_affine(nodes, orig=(-1, 1), dest=(self.stop, self.start))


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

    def __init__(self, intervals: list[Interval]):
        self.intervals = intervals
        self.dimension = len(intervals)
        self.dimensions = tuple(interval.size for interval in self.intervals)

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
            if self.dimension > 1:
                raise IndexError("Invalid index into a Mesh")
            indices = [indices]
        indices = np.asarray(indices)
        # TODO: Type checker complains about the type of this
        return np.stack(
            [self.intervals[n][indices[..., n]] for n in range(self.dimension)], axis=-1
        )

    def to_tensor(self):
        return np.array(list(product(*self.intervals))).reshape(
            *self.dimensions, self.dimension
        )


def array_affine(
    x: np.ndarray,
    orig: tuple,
    dest: tuple,
) -> np.ndarray:
    """
    Performs an affine transformation of x as u = a*x + b from orig=(x0, x1) to dest=(u0, u1).
    """
    # TODO: Maybe combine the affine transformations for vectors, MPS and MPO in a single function?
    x0, x1 = orig
    u0, u1 = dest
    a = (u1 - u0) / (x1 - x0)
    b = 0.5 * ((u1 + u0) - a * (x0 + x1))
    x_affine = a * x
    if abs(b) > np.finfo(np.float64).eps:
        x_affine = x_affine + b
    return x_affine


def mps_to_mesh_matrix(
    sites_per_dimension: list[int], mps_order: str = "A", base: int = 2
) -> np.ndarray:
    """
    Returns a matrix that transforms an array of MPS indices
    to an array of Mesh indices based on the specified order and base.
    """
    if mps_order == "A":
        T = np.zeros((sum(sites_per_dimension), len(sites_per_dimension)), dtype=int)
        start = 0
        for m, n in enumerate(sites_per_dimension):
            T[start : start + n, m] = base ** np.arange(n)[::-1]
            start += n
        return T
    elif mps_order == "B":
        T = np.vstack(
            [
                np.diag(
                    [base ** (n - i - 1) if n > i else 0 for n in sites_per_dimension]
                )
                for i in range(max(sites_per_dimension))
            ]
        )
        T = T[~np.all(T <= 0, axis=1)]
        return T
    else:
        raise ValueError("Invalid MPS order")
