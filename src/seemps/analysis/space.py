import numpy as np
from ..operators import MPO, MPOList, MPOSum


class Space:
    """Class to encode the definition space of a discretized multidimensional function.

    Parameters
    ----------
    qubits_per_dimension : list[int]
        Number of qubits for each dimension.
    L : list[list[floats]]
        Position space intervals [a_i,b_i] for each dimension i.
    closed : bool
        If closed is True, the position space intervals are closed (symmetrically defined).
        If False, the interval is open. (Optional, defaults to True).
    """

    def __init__(self, qubits_per_dimension, L, closed=True):
        self.qubits_per_dimension = qubits_per_dimension
        self.grid_dimensions = [2**n for n in qubits_per_dimension]
        self.closed = closed
        self.n_sites = sum(qubits_per_dimension)
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
        return f"Space(a={self.a}, b={self.b}, dx={self.dx}, closed={self.closed}, qubits={self.qubits_per_dimension})"

    def get_coordinates_tuples(self):
        """Creates a list of coordinates tuples of the form
        (n,k), where n is the dimension and k is the significant digit
        of the qubits used for storing that dimension. Each qubit has
        a tuple (n,k) associated to it.
        """
        coordinates_tuples = []
        coordinates_tuples = [
            (n, k)
            for n, n_q in enumerate(self.qubits_per_dimension)
            for k in range(n_q)
        ]
        return coordinates_tuples

    def get_sites(self):
        """Sites for each dimension"""
        sites = []
        index = 0
        for n in self.qubits_per_dimension:
            sites.append(list(range(index, index + n)))
            index += n
        return sites

    def extend(self, op, dim):
        """Extend MPO acting on 1D to a multi-dimensional MPS."""
        return op.extend(self.n_sites, self.sites[dim])
