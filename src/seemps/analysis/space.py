from __future__ import annotations
from collections.abc import Sequence
import numpy as np
from typing import TypeVar, cast
from ..typing import Real, MPSOrder
from ..operators import MPO, MPOList, MPOSum
from .mesh import Mesh, RegularInterval


# TODO: This might not be the place to have this function
# It should be under operators or in qft.
_Operator = TypeVar("_Operator", MPOSum, MPO, MPOList)


def mpo_flip(operator: _Operator) -> _Operator:
    """Swap the qubits in the quantum register, to fix the reversal
    suffered during the quantum Fourier transform."""
    return operator.reverse()


class Space:
    """Coordinate grid class.

    Class to encode the definition space of a discretized multidimensional function.

    Parameters
    ----------
    qubits_per_dimension : list[int]
        Number of qubits for each dimension.
    L : Sequence[tuple[Real, Real]]
        Position space intervals (a_i,b_i) for each dimension i.
    closed : bool
        If closed is True, the position space intervals are closed (symmetrically defined).
        If False, the interval is open. (defaults to True).
    order : MPSOrder, default = "A"
        The order in which sites are organized. Default is "A" (sequential).
    """

    qubits_per_dimension: list[int]
    grid_dimensions: list[int]
    closed: bool
    n_sites: int
    order: MPSOrder
    sites: list[list[int]]
    L: list[tuple[float, float]]
    mesh: Mesh

    def __init__(
        self,
        qubits_per_dimension: list[int],
        L: Sequence[tuple[Real, Real]],
        closed: bool = True,
        order: MPSOrder = "A",
    ):
        """
        Initializes the Space object.

        Parameters
        ----------
        qubits_per_dimension : list[int]
            Number of qubits for each dimension.
        L : list[list[float]]
            Position space intervals [a_i, b_i] for each dimension i.
        closed : bool, optional
            If True, the intervals are closed; if False, they are open. Default is True.
        order : MPSOrder, optional
            The order in which sites are organized. Default is "A" (sequential).
        """
        self.qubits_per_dimension = qubits_per_dimension
        self.grid_dimensions = [2**n for n in qubits_per_dimension]
        self.closed = closed
        self.n_sites = sum(qubits_per_dimension)
        self.order = order
        self.sites = self.get_sites()
        self.L = [(float(start), float(end)) for start, end in L]
        self.mesh = Mesh(
            [
                RegularInterval(start, end, 2**n, endpoint_right=closed)
                for (start, end), n in zip(self.L, self.qubits_per_dimension)
            ]
        )

    @property
    def dimensions(self) -> int:
        return len(self.qubits_per_dimension)

    @property
    def dx(self) -> np.ndarray:
        return np.asarray([cast(RegularInterval, I).step for I in self.mesh.intervals])

    @property
    def x(self) -> list[np.ndarray]:
        return [I.to_vector() for I in self.mesh.intervals]

    def to_tensor(self) -> np.ndarray:
        return self.mesh.to_tensor().transpose([-1] + list(range(self.dimensions)))

    def change_qubits(self, new_qubits_per_dimension: list[int]) -> Space:
        """
        Creates a new Space object with increased resolution based on the new qubits per dimension.

        Parameters
        ----------
        new_qubits_per_dimension : list[int]
            New number of qubits for each dimension.

        Returns
        -------
        Space
            A new Space object with the increased resolution.
        """
        return Space(
            new_qubits_per_dimension, self.L, closed=self.closed, order=self.order
        )

    def __str__(self):
        """
        Returns a string representation of the Space object.

        Returns
        -------
        str
            String representation of the Space object.
        """
        return f"Space(qubits={self.qubits_per_dimension}, L={self.L}, closed={self.closed}, order={self.order})"

    def get_sites(self):
        """
        Generates the sites for each dimension based on the order.

        Returns
        -------
        list[list[int]]
            A list of lists containing site indices for each dimension.
        """
        sites = []
        index = 0
        if self.order == "A":
            for n in self.qubits_per_dimension:
                sites.append(np.arange(index, index + n).tolist())
                index += n
        else:
            sites = [[] for _ in self.qubits_per_dimension]
            for n in range(max(self.qubits_per_dimension)):
                for d, m in enumerate(self.qubits_per_dimension):
                    if n < m:
                        sites[d].append(index)
                        index += 1
        return sites

    def extend(self, op: _Operator, dim: int) -> _Operator:
        """
        Extends an MPO acting on a 1D space to a multi-dimensional MPS.

        Parameters
        ----------
        op : MPO
            The MPO to extend.
        dim : int
            The dimension to extend along.

        Returns
        -------
        MPO
            The extended multi-dimensional MPO.
        """
        return op.extend(self.n_sites, self.sites[dim])

    def enlarge_dimension(self, dim: int, amount: int) -> Space:
        """
        Enlarges the specified dimension by adding more qubits to one dimension.

        Parameters
        ----------
        dim : int
            The dimension to enlarge.
        amount : int
            The number of qubits to add.

        Returns
        -------
        Space
            A new Space object with the enlarged dimension.
        """
        new_qubits_per_dimension = self.qubits_per_dimension.copy()
        new_qubits_per_dimension[dim] += amount
        return Space(new_qubits_per_dimension, self.L, self.closed, self.order)

    def new_positions_from_old_space(self, space: Space) -> list[int]:
        """
        Maps the qubits from a smaller space, to their respective positions
        in the quantum register for a larger space.

        Parameters
        ----------
        space : Space
            The old Space object to map positions from.

        Returns
        -------
        list[int]
            List of new positions in the current Space object.
        """
        new_positions = self.sites.copy()
        for d, n in enumerate(space.qubits_per_dimension):
            if n > self.qubits_per_dimension[d]:
                raise Exception(
                    f"I cannot map a larger map into a smaller one.\nOld: {space}\nNew: {self}"
                )
            new_positions[d] = new_positions[d][:n]
        return sorted(sum(new_positions, []))
