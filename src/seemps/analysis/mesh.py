from __future__ import annotations
from abc import ABC, abstractmethod
from itertools import product
from collections.abc import Sequence, Iterator
from typing import overload
import numpy as np
from numpy.typing import ArrayLike, NDArray
from ..typing import Vector, Matrix


class Interval(ABC):
    """
    Interval Abstract Base Class.

    This class represents implicitly a univariate discretization along `size`
    points within two endpoints `start` and `stop`. The elements of an `Interval`
    can be indexed as in `i[0]`, `i[1]`,... up to `i[size-1]` and they can
    be converted to other sequences, as in `list(i)`, or iterated over.

    Parameters
    ----------
    start : float
        The initial point of the interval.
    stop : float
        The ending point of the interval.
    size : int
        The discretization size, i.e. number of points of the interval within
        `start` and `stop`.
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

    def _validate_index(self, idx: int | np.ndarray):
        if isinstance(idx, int):
            if not (0 <= idx < self.size):
                raise IndexError("Index out of range")
        elif isinstance(idx, np.ndarray):
            if not np.all((0 <= idx) & (idx < self.size)):
                raise IndexError("Index out of range")
        else:
            raise TypeError("Index must be an integer or a NumPy array")

    @overload
    def __getitem__(self, idx: NDArray[np.integer]) -> NDArray[np.floating]: ...

    @overload
    def __getitem__(self, idx: int) -> float: ...

    @abstractmethod
    def __getitem__(
        self, idx: int | NDArray[np.integer]
    ) -> float | NDArray[np.floating]: ...

    def to_vector(self) -> np.ndarray:
        return np.array([self[idx] for idx in range(self.size)])

    def map_to(self, start: float, stop: float) -> Interval:
        return type(self)(start, stop, self.size)

    def update_size(self, size: int) -> Interval:
        return type(self)(self.start, self.stop, size)

    def __iter__(self) -> Iterator:
        return (self[i] for i in range(self.size))


class IntegerInterval(Interval):
    """Equispaced integer discretization between `start` and `stop` with given `step`."""

    step: int

    def __init__(self, start: int, stop: int, step: int = 1):
        self.step = step
        size = (stop - start + step - 1) // step
        super().__init__(start, stop, size)

    @overload
    def __getitem__(self, idx: NDArray[np.integer]) -> NDArray[np.floating]: ...

    @overload
    def __getitem__(self, idx: int) -> float: ...

    def __getitem__(
        self, idx: int | NDArray[np.integer]
    ) -> float | NDArray[np.floating]:
        super()._validate_index(idx)
        return self.start + idx * self.step


class RegularInterval(Interval):
    """
    Equispaced discretization between `start` and `stop` with `size` points.
    The left and right boundary conditions can be set open or closed by
    respectively setting the `endpoint_right` and `endpoint_left` flags.
    Defaults to a closed-left, open-right interval [start, stop).
    """

    endpoint_left: bool
    endpoint_right: bool
    num_steps: int
    step: float
    start_displaced: float

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
    def __getitem__(self, idx: NDArray[np.integer]) -> NDArray[np.floating]: ...

    @overload
    def __getitem__(self, idx: int) -> float: ...

    def __getitem__(
        self, idx: int | NDArray[np.integer]
    ) -> float | NDArray[np.floating]:
        super()._validate_index(idx)
        return self.start_displaced + idx * self.step


class ChebyshevInterval(Interval):
    """
    Irregular discretization between `start` and `stop` given by the zeros or extrema
    of a Chebyshev polynomial of order `size` or `size-1` respectively.
    The nodes are affinely transformed from the canonical [-1, 1] interval to [start, stop].
    If `endpoints` is set, returns the Chebyshev extrema, defined in the closed interval [a, b].
    Else, returns the Chebyshev zeros defined in the open interval (start, stop).
    """

    endpoints: bool

    def __init__(self, start: float, stop: float, size: int, endpoints: bool = False):
        super().__init__(start, stop, size)
        self.endpoints = endpoints

    @overload
    def __getitem__(self, idx: NDArray[np.integer]) -> NDArray[np.floating]: ...

    @overload
    def __getitem__(self, idx: int) -> float: ...

    def __getitem__(
        self, idx: int | NDArray[np.integer]
    ) -> float | NDArray[np.floating]:
        super()._validate_index(idx)
        if self.endpoints:  # Chebyshev extrema
            nodes = np.cos(np.pi * idx / (self.size - 1))
        else:  # Chebyshev zeros
            nodes = np.cos(np.pi * (2 * idx + 1) / (2 * self.size))
        return array_affine(nodes, orig=(-1, 1), dest=(self.stop, self.start))


class ArrayInterval(Interval):
    """Wrapper class that allows passing an explicit 1D array of values as an Interval."""

    def __init__(self, array: np.ndarray):
        if array.ndim != 1:
            raise ValueError("ArrayInterval requires a 1D array of floats")
        self.values = np.asarray(array, float)
        super().__init__(self.values[0], self.values[-1], len(self.values))

    @overload
    def __getitem__(self, idx: NDArray[np.integer]) -> NDArray[np.floating]: ...
    @overload
    def __getitem__(self, idx: int) -> float: ...

    def __getitem__(
        self, idx: int | NDArray[np.integer]
    ) -> float | NDArray[np.floating]:
        self._validate_index(idx)
        return self.values[idx]

    def to_vector(self) -> np.ndarray:
        return self.values

    def update_size(self, size: int) -> ArrayInterval:
        raise NotImplementedError("ArrayInterval does not support update_size.")

    def map_to(self, start: float, stop: float) -> ArrayInterval:
        array = array_affine(self.values, (self.start, self.stop), (start, stop))
        return ArrayInterval(array)


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
    dimensions : tuple[int]
        Tuple of the sizes of each interval
    """

    intervals: list[Interval]
    dimension: int
    dimensions: tuple[int, ...]

    def __init__(self, intervals: list[Interval]):
        self.intervals = intervals
        self.dimension = len(intervals)
        self.dimensions = tuple(interval.size for interval in self.intervals)

    def __getitem__(
        self, indices: int | Sequence[int] | ArrayLike
    ) -> NDArray[np.floating]:
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
        indices : np.ndarray
            An integer, or an array-like structure indexing points
            in the mesh.

        Returns
        -------
        points : np.ndarray[float]
            Coordinates of one or more points.
        """
        if isinstance(indices, int):
            if self.dimension > 1:
                raise IndexError("Invalid index into a Mesh")
            indices = [indices]
        index_array = np.asarray(indices)
        return np.stack(
            [self.intervals[n][index_array[..., n]] for n in range(self.dimension)],
            axis=-1,
        )

    def displace(self, displacement: Vector) -> Mesh:
        if len(displacement) != self.dimension:
            raise ValueError("Displacement size does not match mesh dimension")
        displaced_intervals = []
        for k, interval in enumerate(self.intervals):
            a, b, x = interval.start, interval.stop, displacement[k]
            displaced_intervals.append(interval.map_to(a - x, b - x))
        return Mesh(displaced_intervals)

    def to_tensor(self, channels_first: bool = False) -> NDArray[np.floating]:
        """
        Converts the mesh object to a tensor by computing the tensor product of the intervals.

        Parameters
        ----------
        channels_first: bool, default=True
            Whether to set the dimension index 'm' as the first or the last index or the tensor.
        """
        tensor = np.array(list(product(*self.intervals))).reshape(
            *self.dimensions, self.dimension
        )
        return np.moveaxis(tensor, -1, 0) if channels_first else tensor


def array_affine(
    array: np.ndarray,
    orig: tuple,
    dest: tuple,
) -> np.ndarray:
    """
    Performs an affine transformation of a given `array` as u = a*x + b from orig=(x0, x1) to dest=(u0, u1).
    """
    x0, x1 = orig
    u0, u1 = dest
    a = (u1 - u0) / (x1 - x0)
    b = 0.5 * ((u1 + u0) - a * (x0 + x1))
    x_affine = a * array
    if abs(b) > np.finfo(np.float64).eps:
        x_affine = x_affine + b
    return x_affine


def mps_to_mesh_matrix(
    sites_per_dimension: list[int], permutation: Vector | None = None, base: int = 2
) -> Matrix:
    """
    Returns a transformation matrix T that maps between MPS indices and multi-dimensional mesh coordinates.

    For a mesh with m dimensions and n = `sum(sites_per_dimension)` sites, T is a matrix of shape
    (n, m). Row r corresponds to an MPS site while column i contains the contribution of that site
    to dimension i, such that the integer mesh coordinates read:

        x_i = sum_r physical_indices[r] * T[r, i].

    Each dimension i uses `sites_per_dimension[i]` consecutive sites given by decreasing powers of `base`.
    If `permutation` is provided, the rows of T are reordered accordinglyy.

    Parameters
    ----------
    sites_per_dimension : list[int]
        Number of sites allocated to each dimension.
    permutation : Vector | None
        Optional row permutation. Defaults to None (no permutation).
    base : int
        Local physical dimension per site.

    Returns
    -------
    Matrix
        Linear mapping of shape (N, m) with integer `base^k` weights.

    Examples
    --------
    sites_per_dimension = [2, 3] with base 2 yields:

        T = [[2, 0],   # site 0 → x contributes base^1
             [1, 0],   # site 1 → x contributes base^0
             [0, 4],   # site 2 → y contributes base^2
             [0, 2],   # site 3 → y contributes base^1
             [0, 1]]   # site 4 → y contributes base^0
    """
    m = len(sites_per_dimension)
    n_total = sum(sites_per_dimension)

    offset = 0
    T = np.zeros((n_total, m), dtype=int)
    for i, n_i in enumerate(sites_per_dimension):
        for j in range(n_i):
            row = offset + j
            T[row, i] = base ** (n_i - 1 - j)
        offset += n_i

    return T[permutation] if permutation is not None else T


def interleaving_permutation(sites_per_dimension: list[int]) -> Vector:
    """
    Return a permutation vector that interleaves MPS sites across dimensions, taking one site from
    each dimension in order of increasing significance ("B" order).
    This permutation groups together bits from all dimensions by significance.
    """
    m = len(sites_per_dimension)
    n_max = max(sites_per_dimension)

    offsets = np.cumsum([0] + sites_per_dimension[:-1])
    permutation = []
    for i in range(n_max):
        for j in range(m):
            if i < sites_per_dimension[j]:
                permutation.append(offsets[j] + i)

    return np.array(permutation, dtype=int)
