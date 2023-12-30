import numpy as np
from .tools import TestCase
from seemps.analysis.mesh import *


class TestIntervals(TestCase):
    def test_regular_closed_interval_constructor(self):
        I = RegularClosedInterval(0, 1, 3)
        self.assertEqual(I.start, 0)
        self.assertEqual(I.stop, 1)
        self.assertEqual(I.size, 3)
        self.assertEqual(len(I), 3)
        self.assertEqual(I[0], 0)
        self.assertEqual(I[1], 0.5)
        self.assertEqual(I[2], 1.0)

        self.assertEqual([I[0], I[1], I[2]], list(I))
        self.assertEqual([I[0], I[1], I[2]], [x for x in I])

    def test_regular_half_open_interval_constructor(self):
        I = RegularHalfOpenInterval(0, 1, 2)
        self.assertEqual(I.start, 0)
        self.assertEqual(I.stop, 1)
        self.assertEqual(I.size, 2)
        self.assertEqual(len(I), 2)
        self.assertEqual(I[0], 0)
        self.assertEqual(I[1], 0.5)

        self.assertEqual([I[0], I[1]], list(I))
        self.assertEqual([I[0], I[1]], [x for x in I])

    def test_regular_chebyshev_zeros_interval_constructor(self):
        I = ChebyshevZerosInterval(-1, 1, 2)
        self.assertEqual(I.start, -1)
        self.assertEqual(I.stop, 1)
        self.assertEqual(I.size, 2)
        self.assertEqual(len(I), 2)
        self.assertAlmostEqual(I[0], -np.sqrt(2.0) / 2.0)
        self.assertAlmostEqual(I[1], np.sqrt(2.0) / 2.0)

        self.assertEqual([I[0], I[1]], list(I))
        self.assertEqual([I[0], I[1]], [x for x in I])
