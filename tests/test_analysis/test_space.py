import numpy as np
from seemps.analysis.space import Space
from ..tools import *


class TestSpace(TestCase):
    qubits = [[3], [3, 3], [2, 4, 3]]

    def test_coordinates(self):
        a, b = 0, 1
        for qubits_i in self.qubits:
            for closed in [False, True]:
                dims = [2**q for q in qubits_i]
                L = [[a, b]] * len(qubits_i)
                dx = np.array(
                    [
                        (end - start) / ((d - 1) if closed else d)
                        for (start, end), d in zip(L, dims)
                    ]
                )
                space = Space(qubits_i, L, closed=closed)
                for i, dim in enumerate(dims):
                    x = a + dx[i] * np.arange(dim)
                    self.assertSimilar(x, space.x[i])

    def test_increase_resolution(self):
        a, b = 0, 1
        for qubits_i in self.qubits:
            for closed in [False, True]:
                dims = [2**q for q in qubits_i]
                L = [[a, b]] * len(qubits_i)
                dx = np.array(
                    [
                        (end - start) / ((d - 1) if closed else d)
                        for (start, end), d in zip(L, dims)
                    ]
                )
                space = Space(qubits_i, L, closed=closed)
                new_qubits = [q + 1 for q in qubits_i]
                new_space = space.increase_resolution(new_qubits)
                new_dims = [2**q for q in new_qubits]
                new_dx = [dx * dims[i] / new_dims[i] for i, dx in enumerate(space.dx)]
                for i, dim in enumerate(new_dims):
                    x = a + new_dx[i] * np.arange(dim)
                    self.assertSimilar(x, new_space.x[i])
