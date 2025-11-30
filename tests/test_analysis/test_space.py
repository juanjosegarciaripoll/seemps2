import numpy as np
from seemps.analysis.space import Space
from ..tools import TestCase


class TestSpace(TestCase):
    qubits = [[3], [3, 3], [2, 4, 3]]

    def test_space_three_qubits_open_1d(self):
        space = Space([3], L=[(0, 1)], closed=False)
        self.assertEqual(space.qubits_per_dimension, [3])
        self.assertEqual(space.grid_dimensions, [2**3])
        self.assertEqual(space.n_sites, 3)
        dx = 1.0 / 8
        self.assertSimilar(space.dx, np.asarray([dx]))
        self.assertEqual(space.x[0][0], 0.0)
        self.assertEqual(space.x[0][-1], 1.0 - dx)
        self.assertSimilar(space.x, [np.arange(8) * dx])
        self.assertEqual(space.L, [(0, 1)])
        self.assertEqual(space.get_sites(), [[0, 1, 2]])
        self.assertEqual(space.order, "A")

    def test_space_three_qubits_closed_1d(self):
        space = Space([3], L=[(0, 1)], closed=True)
        self.assertEqual(space.qubits_per_dimension, [3])
        self.assertEqual(space.grid_dimensions, [2**3])
        self.assertEqual(space.n_sites, 3)
        dx = 1.0 / 7
        self.assertSimilar(space.dx, np.asarray([dx]))
        self.assertEqual(space.x[0][0], 0.0)
        self.assertEqual(space.x[0][-1], 1.0)
        self.assertSimilar(space.x, [np.arange(8) * dx])
        self.assertEqual(space.L, [(0, 1)])
        self.assertEqual(space.get_sites(), [[0, 1, 2]])
        self.assertEqual(space.order, "A")

    def test_space_open_2d(self):
        space = Space([3, 2], L=[(0, 1), (0, 1)], closed=False)
        self.assertEqual(space.qubits_per_dimension, [3, 2])
        self.assertEqual(space.grid_dimensions, [2**3, 2**2])
        self.assertEqual(space.n_sites, 3 + 2)
        self.assertSimilar(space.dx, np.asarray([1.0 / 8, 1.0 / 4]))
        self.assertEqual(space.x[0][0], 0.0)
        self.assertEqual(space.x[0][-1], 1.0 - 1.0 / 8)
        self.assertEqual(space.x[1][0], 0.0)
        self.assertEqual(space.x[1][-1], 1.0 - 1.0 / 4)
        self.assertEqual(len(space.x), 2)
        self.assertSimilar(space.x[0], np.arange(8) * space.dx[0])
        self.assertSimilar(space.x[1], np.arange(4) * space.dx[1])
        self.assertEqual(space.L, [(0, 1), (0, 1)])
        self.assertEqual(space.get_sites(), [[0, 1, 2], [3, 4]])
        self.assertEqual(space.order, "A")

    def test_space_closed_2d(self):
        space = Space([3, 2], L=[(0, 1), (-1, 1)], closed=True, order="B")
        self.assertEqual(space.qubits_per_dimension, [3, 2])
        self.assertEqual(space.grid_dimensions, [2**3, 2**2])
        self.assertEqual(space.n_sites, 3 + 2)
        self.assertSimilar(space.dx, np.asarray([1.0 / 7, 2.0 / 3]))
        self.assertEqual(space.x[0][0], 0.0)
        self.assertEqual(space.x[0][-1], 1.0)
        self.assertEqual(space.x[1][0], -1.0)
        self.assertEqual(space.x[1][-1], 1.0)
        self.assertEqual(len(space.x), 2)
        self.assertSimilar(space.x[0], np.arange(8) * space.dx[0])
        self.assertSimilar(space.x[1], -1.0 + np.arange(4) * space.dx[1])
        self.assertEqual(space.L, [(0, 1), (-1, 1)])
        self.assertEqual(space.get_sites(), [[0, 2, 4], [1, 3]])
        self.assertEqual(space.order, "B")

    def test_space_increase_resolution(self):
        space = Space([2, 3], L=[(0, 1), (1, 3)], closed=True)
        new_space = space.change_qubits([4, 2])
        self.assertEqual(new_space.qubits_per_dimension, [4, 2])
        self.assertEqual(new_space.grid_dimensions, [2**4, 2**2])
        self.assertSimilar(new_space.dx, [1.0 / 15, 2.0 / 3])
        self.assertEqual(new_space.x[0][0], 0.0)
        self.assertEqual(new_space.x[0][-1], 1.0)
        self.assertEqual(new_space.x[1][0], 1.0)
        self.assertEqual(new_space.x[1][-1], 3.0)

    def test_space_new_positions_A_order(self):
        space = Space([2, 3], L=[(0, 1), (1, 3)], closed=True)
        new_space = space.change_qubits([4, 4])
        self.assertEqual(new_space.get_sites(), [[0, 1, 2, 3], [4, 5, 6, 7]])
        self.assertEqual(new_space.new_positions_from_old_space(space), [0, 1, 4, 5, 6])

    def test_space_new_positions_B_order(self):
        space = Space([2, 3], L=[(0, 1), (1, 3)], closed=True, order="B")
        new_space = space.change_qubits([4, 4])
        self.assertEqual(space.get_sites(), [[0, 2], [1, 3, 4]])
        self.assertEqual(new_space.get_sites(), [[0, 2, 4, 6], [1, 3, 5, 7]])
        self.assertEqual(new_space.new_positions_from_old_space(space), [0, 1, 2, 3, 5])
