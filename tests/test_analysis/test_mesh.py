import numpy as np
from math import sqrt
from seemps.analysis.mesh import (
    Mesh,
    RegularInterval,
    ChebyshevInterval,
    mps_to_mesh_matrix,
    interleaving_permutation,
)
from ..tools import TestCase


class TestIntervals(TestCase):
    def test_regular_closed_interval_constructor(self):
        I = RegularInterval(0, 1, 3, endpoint_right=True)
        self.assertEqual(I.start, 0)
        self.assertEqual(I.stop, 1)
        self.assertEqual(I.size, 3)
        self.assertEqual(len(I), 3)
        self.assertEqual(I[0], 0)
        self.assertEqual(I[1], 0.5)
        self.assertEqual(I[2], 1.0)

        self.assertEqual([I[0], I[1], I[2]], list(I))
        self.assertEqual([I[0], I[1], I[2]], [x for x in I])

    def test_regular_closed_open_interval_constructor(self):
        I = RegularInterval(0, 1, 2)
        self.assertEqual(I.start, 0)
        self.assertEqual(I.stop, 1)
        self.assertEqual(I.size, 2)
        self.assertEqual(len(I), 2)
        self.assertEqual(I[0], 0)
        self.assertEqual(I[1], 0.5)

        self.assertEqual([I[0], I[1]], list(I))
        self.assertEqual([I[0], I[1]], [x for x in I])

    def test_regular_open_closed_interval_constructor(self):
        I = RegularInterval(0, 1, 2, endpoint_left=False, endpoint_right=True)
        self.assertEqual(I.start, 0)
        self.assertEqual(I.stop, 1)
        self.assertEqual(I.size, 2)
        self.assertEqual(len(I), 2)
        self.assertEqual(I[0], 0.5)
        self.assertEqual(I[1], 1)

        self.assertEqual([I[0], I[1]], list(I))
        self.assertEqual([I[0], I[1]], [x for x in I])

    def test_regular_open_interval_constructor(self):
        I = RegularInterval(0, 1, 3, endpoint_left=False)
        self.assertEqual(I.start, 0)
        self.assertEqual(I.stop, 1)
        self.assertEqual(I.size, 3)
        self.assertEqual(len(I), 3)
        self.assertEqual(I[0], 0.25)
        self.assertEqual(I[1], 0.5)
        self.assertEqual(I[2], 0.75)

        self.assertEqual([I[0], I[1], I[2]], list(I))
        self.assertEqual([I[0], I[1], I[2]], [x for x in I])

    def test_regular_chebyshev_zeros_interval_constructor(self):
        I = ChebyshevInterval(-1, 1, 2)
        self.assertEqual(I.start, -1)
        self.assertEqual(I.stop, 1)
        self.assertEqual(I.size, 2)
        self.assertEqual(len(I), 2)
        self.assertAlmostEqual(I[0], -sqrt(2.0) / 2.0)
        self.assertAlmostEqual(I[1], sqrt(2.0) / 2.0)

        self.assertEqual([I[0], I[1]], list(I))
        self.assertEqual([I[0], I[1]], [x for x in I])

    def test_regular_chebyshev_extrema_interval_constructor(self):
        I = ChebyshevInterval(-1, 1, 2, endpoints=True)
        self.assertEqual(I.start, -1)
        self.assertEqual(I.stop, 1)
        self.assertEqual(I.size, 2)
        self.assertEqual(len(I), 2)
        self.assertAlmostEqual(I[0], -1)
        self.assertAlmostEqual(I[1], 1)

        self.assertEqual([I[0], I[1]], list(I))
        self.assertEqual([I[0], I[1]], [x for x in I])

    def test_rescaled_chebyshev_zeros_interval(self):
        f = lambda x: 2 * x + 2  # noqa: E731
        I = ChebyshevInterval(f(-1), f(1), 2)
        self.assertEqual(I.start, f(-1))
        self.assertEqual(I.stop, f(1))
        self.assertAlmostEqual(I[0], f(-sqrt(2.0) / 2.0))
        self.assertAlmostEqual(I[1], f(sqrt(2.0) / 2.0))


class TestMesh(TestCase):
    def test_mesh_constructor_1d(self):
        I0 = RegularInterval(0, 1, 3, endpoint_right=True)
        m = Mesh([I0])
        self.assertEqual(len(m.intervals), 1)
        self.assertEqual(m.intervals[0], I0)
        self.assertEqual(m.dimension, 1)
        self.assertEqual(m.dimensions, (3,))

    def test_mesh_1d_sequence_access(self):
        m = Mesh([RegularInterval(0, 1, 3, endpoint_right=True)])
        self.assertEqual(m[[0]], 0.0)
        self.assertEqual(m[[1]], 0.5)
        self.assertEqual(m[[2]], 1.0)
        with self.assertRaises(IndexError):
            m[[3]]

    def test_mesh_1d_integer_access(self):
        m = Mesh([RegularInterval(0, 1, 3, endpoint_right=True)])
        self.assertEqual(m[0], 0.0)
        self.assertEqual(m[1], 0.5)
        self.assertEqual(m[2], 1.0)

    def test_mesh_1d_checks_bounds(self):
        m = Mesh([RegularInterval(0, 1, 3, endpoint_right=True)])
        with self.assertRaises(IndexError):
            m[3]
        with self.assertRaises(IndexError):
            m[3, 0]

    def test_mesh_1d_multiple_index_access(self):
        m = Mesh([RegularInterval(0, 1, 3, endpoint_right=True)])
        self.assertSimilar(m[[[0], [2], [1]]], [[0.0], [1.0], [0.5]])

    def test_mesh_1d_to_tensor(self):
        m = Mesh([RegularInterval(0, 1, 3, endpoint_right=True)])
        self.assertSimilar(m.to_tensor(), [(0,), (0.5,), (1.0,)])

    def test_mesh_constructor_2d(self):
        I0 = RegularInterval(0, 1, 3, endpoint_right=True)
        I1 = RegularInterval(0, 1, 2)
        m = Mesh([I0, I1])
        self.assertEqual(len(m.intervals), 2)
        self.assertEqual(m.intervals[0], I0)
        self.assertEqual(m.intervals[1], I1)
        self.assertEqual(m.dimension, 2)
        self.assertEqual(m.dimensions, (3, 2))

    def test_mesh_2d_tuple_access(self):
        m = Mesh(
            [
                RegularInterval(0, 1, 3, endpoint_right=True),
                RegularInterval(0, 1, 2),
            ]
        )
        self.assertSimilar(m[0, 0], [0.0, 0.0])
        self.assertSimilar(m[1, 0], [0.5, 0.0])
        self.assertSimilar(m[0, 1], [0.0, 0.5])
        self.assertSimilar(m[1, 1], [0.5, 0.5])
        self.assertSimilar(m[2, 0], [1.0, 0.0])
        self.assertSimilar(m[2, 1], [1.0, 0.5])

    def test_mesh_2d_multiple_index_access(self):
        m = Mesh(
            [
                RegularInterval(0, 1, 3, endpoint_right=True),
                RegularInterval(0, 1, 2),
            ]
        )
        self.assertSimilar(
            m[[(0, 0), (1, 0), (0, 1), (2, 0), (1, 1)]],
            [(0.0, 0.0), (0.5, 0.0), (0.0, 0.5), (1.0, 0.0), (0.5, 0.5)],
        )

    def test_mesh_2d_checks_bounds(self):
        m = Mesh(
            [RegularInterval(0, 1, 3, endpoint_right=True), RegularInterval(0, 1, 2)]
        )
        with self.assertRaises(IndexError):
            m[3, 0]
        with self.assertRaises(IndexError):
            m[0]

    def test_mesh_2d_to_tensor(self):
        m = Mesh(
            [RegularInterval(0, 1, 3, endpoint_right=True), RegularInterval(0, 1, 2)]
        )
        self.assertSimilar(
            m.to_tensor(),
            [
                [(0.0, 0.0), (0.0, 0.5)],
                [(0.5, 0.0), (0.5, 0.5)],
                [(1.0, 0.0), (1.0, 0.5)],
            ],
        )

    def test_mesh_transformation_matrix_A_order(self):
        T = mps_to_mesh_matrix([1])
        self.assertSimilar(T, np.eye(1))

        T = mps_to_mesh_matrix([2])
        self.assertSimilar(T, [[2], [1]])

        T = mps_to_mesh_matrix([1, 1])
        self.assertSimilar(T, np.eye(2))

        T = mps_to_mesh_matrix([1, 2])
        self.assertSimilar(T, [[1.0, 0.0], [0.0, 2.0], [0.0, 1.0]])

        T = mps_to_mesh_matrix([2, 2])
        self.assertSimilar(T, [[2.0, 0.0], [1.0, 0.0], [0.0, 2.0], [0.0, 1.0]])

    def test_mesh_transformation_matrix_B_order(self):
        T = mps_to_mesh_matrix([1], interleaving_permutation([1]))
        self.assertSimilar(T, np.eye(1))

        T = mps_to_mesh_matrix([2], interleaving_permutation([2]))
        self.assertSimilar(T, [[2], [1]])

        T = mps_to_mesh_matrix([1, 1], interleaving_permutation([1, 1]))
        self.assertSimilar(T, np.eye(2))

        T = mps_to_mesh_matrix([1, 2], interleaving_permutation([1, 2]))
        self.assertSimilar(T, [[1.0, 0.0], [0.0, 2.0], [0.0, 1.0]])

        T = mps_to_mesh_matrix([2, 2], interleaving_permutation([2, 2]))
        self.assertSimilar(T, [[2.0, 0.0], [0.0, 2.0], [1.0, 0.0], [0.0, 1.0]])
