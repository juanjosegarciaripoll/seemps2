import numpy as np

from seemps.analysis.factories import mps_sin
from seemps.analysis.optimization import optimize_mps

from ..tools import TestCase


def _bits_to_int(bits: np.ndarray) -> int:
    return int("".join(str(x) for x in bits), 2)


class TestMPSOptimization(TestCase):
    def test_optimize_mps_sin(self):
        mps = mps_sin(-2, 2, 6)
        # Vector optimization
        y = mps.to_vector()
        i_min_vec = np.argmin(y)
        y_min_vec = y[i_min_vec]
        i_max_vec = np.argmax(y)
        y_max_vec = y[i_max_vec]
        # MPS optimization
        (j_min, y_min), (j_max, y_max) = optimize_mps(mps)
        i_min = _bits_to_int(j_min)
        i_max = _bits_to_int(j_max)
        # TODO: There is an error of 10**(-16) of unknown source.
        self.assertEqual(i_min_vec, i_min)
        self.assertAlmostEqual(y_min_vec, y_min, places=15)
        self.assertEqual(i_max_vec, i_max)
        self.assertAlmostEqual(y_max_vec, y_max, places=15)
