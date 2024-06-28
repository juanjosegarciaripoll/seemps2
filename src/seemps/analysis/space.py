from __future__ import annotations
import numpy as np
from ..operators import MPO, MPOList, MPOSum


# TODO: This might not be the place to have this function
# It should be under operators or in qft.
def mpo_flip(operator):
    """Swap the qubits in the quantum register, to fix the reversal
    suffered during the quantum Fourier transform."""
    if isinstance(operator, MPO):
        return MPO(
            [np.moveaxis(op, [0, 1, 2, 3], [3, 1, 2, 0]) for op in reversed(operator)],
            strategy=operator.strategy,
        )
    elif isinstance(operator, MPOList):
        return MPOList(
            [
                MPO(
                    [
                        np.moveaxis(op, [0, 1, 2, 3], [3, 1, 2, 0])
                        for op in reversed(mpo)
                    ],
                    strategy=operator.strategy,
                )
                for mpo in operator.mpos
            ],
            strategy=operator.strategy,
        )
    elif isinstance(operator, MPOSum):
        new_mpos = []
        for weight, op in zip(operator.weights, operator.mpos):
            new_mpos.append(weight * mpo_flip(op))
        return MPOSum(
            new_mpos,
            strategy=operator.strategy,
        )


class Space:
    """Coordinate grid class.
    
    Class to encode the definition space of a discretized multidimensional function.

    Parameters
    ----------
    qubits_per_dimension : list[int]
        Number of qubits for each dimension.
    L : list[list[floats]]
        Position space intervals [a_i,b_i] for each dimension i.
    closed : bool
        If closed is True, the position space intervals are closed (symmetrically defined).
        If False, the interval is open. (Optional, defaults to True).
    order : str, optional
            The order in which sites are organized. Default is "A" (sequential).
    """

    def __init__(self, qubits_per_dimension, L, closed=True, order="A"):
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
        order : str, optional
            The order in which sites are organized. Default is "A" (sequential).
        """
        self.qubits_per_dimension = qubits_per_dimension
        self.grid_dimensions = [2**n for n in qubits_per_dimension]
        self.closed = closed
        self.n_sites = sum(qubits_per_dimension)
        self.order = order
        self.sites = self.get_sites()
        self.L = L
        self.a = [L_i[0] for L_i in L]
        self.b = [L_i[1] for L_i in L]
        self.dx = np.array(
            [
                (end - start) / ((d - 1) if closed else d)
                for (start, end), d in zip(L, self.grid_dimensions)
            ]
        )
        self.x = [
            self.a[i] + self.dx[i] * np.arange(dim)
            for i, dim in enumerate(self.grid_dimensions)
        ]

    def increase_resolution(self, new_qubits_per_dimension):
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
        if self.closed:
            new_space = Space(
                new_qubits_per_dimension,
                self.L,
                closed=self.closed,
            )
            new_space.dx = np.array(
                [
                    dx * self.grid_dimensions[i] / new_space.grid_dimensions[i]
                    for i, dx in enumerate(self.dx)
                ]
            )
            new_space.x = [
                new_space.a[i] + new_space.dx[i] * np.arange(dim)
                for i, dim in enumerate(new_space.grid_dimensions)
            ]
        else:
            new_space = Space(
                new_qubits_per_dimension,
                [
                    (an, an + dxn * (2**old_qubits))
                    for an, dxn, old_qubits in zip(
                        self.a, self.dx, self.qubits_per_dimension
                    )
                ],
                closed=self.closed,
            )
        return new_space

    def __str__(self):
        """
        Returns a string representation of the Space object.

        Returns
        -------
        str
            String representation of the Space object.
        """
        return f"Space(a={self.a}, b={self.b}, dx={self.dx}, closed={self.closed}, qubits={self.qubits_per_dimension})"

    def get_coordinates_tuples(self):
        """
        Creates a list of coordinate tuples for the qubits.

        Returns
        -------
        list[tuple]
            List of tuples (n, k), where `n` is the dimension and `k` is the significant digit 
            of the qubits used for storing that dimension.
        """
        coordinates_tuples = []
        coordinates_tuples = [
            (n, k)
            for n, n_q in enumerate(self.qubits_per_dimension)
            for k in range(n_q)
        ]
        return coordinates_tuples

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

    def extend(self, op, dim):
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
        MPS
            The extended multi-dimensional MPS.
        """
        return op.extend(self.n_sites, self.sites[dim])

    def enlarge_dimension(self, dim, amount) -> Space:
        """
        Enlarges the specified dimension by adding more qubits.

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
        Maps new positions from an old Space object to the current Space object.

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
            new_positions[d] = new_positions[d][:n]
        new_positions = sum(new_positions, [])
        new_positions.sort()
        return new_positions
