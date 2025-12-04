import numpy as np

from seemps.state import MPS
from seemps.analysis.factories import mps_sin
from seemps.analysis.mesh import RegularInterval
from seemps.analysis.optimization import (
    optimize_mps,
    binary_search_mps,
    get_search_environments,
)

from ..tools import TestCase


def _bits_to_int(bits: np.ndarray) -> int:
    return int("".join(str(x) for x in bits), 2)


class TestBinarySearch(TestCase):
    def _make_monotone_mps(self, f, a=-2, b=2, n=7):
        interval = RegularInterval(a, b, 2**n)
        x = interval.to_vector()
        y = f(x)
        dimensions = [2] * n
        return MPS.from_vector(y, dimensions, normalize=False), x, y

    def test_binary_search_increasing(self):
        f = lambda x: x**3  # noqa: E731
        a, b, n = -1, 1, 10
        mps, _, y = self._make_monotone_mps(f, a, b, n)

        rng = np.random.default_rng(42)
        thresholds = rng.uniform(a, b, size=10)
        for t in thresholds:
            envs = get_search_environments(mps)
            bits_cached = binary_search_mps(mps, t, search_environments=envs)
            bits_fresh = binary_search_mps(mps, t)
            self.assertSimilar(bits_cached, bits_fresh)

            idx = int("".join(str(b) for b in bits_cached), 2)
            idx_ref = np.argmax(y > t)
            self.assertEqual(idx, idx_ref)

    def test_binary_search_decreasing(self):
        f = lambda x: -(x**3)  # noqa: E731
        a, b, n = -1, 1, 10
        mps, _, y = self._make_monotone_mps(f, a, b, n)

        rng = np.random.default_rng(42)
        thresholds = rng.uniform(a, b, size=10)
        for t in thresholds:
            envs = get_search_environments(mps)
            bits_cached = binary_search_mps(
                mps, t, increasing=False, search_environments=envs
            )
            bits_fresh = binary_search_mps(mps, t, increasing=False)
            self.assertSimilar(bits_cached, bits_fresh)

            idx = int("".join(str(b) for b in bits_cached), 2)
            idx_ref = np.argmax(y <= t)
            self.assertEqual(idx, idx_ref)


class TestMPSOptimization(TestCase):
    def test_optimize_mps_sin(self):
        mps = mps_sin(-2, 2, 6)
        y = mps.to_vector()
        i_min, i_max = np.argmin(y), np.argmax(y)
        (bits_min, z_min), (bits_max, z_max) = optimize_mps(mps)
        j_min, j_max = _bits_to_int(bits_min), _bits_to_int(bits_max)
        self.assertEqual(i_min, j_min)
        self.assertAlmostEqual(y[i_min], z_min, places=15)
        self.assertEqual(i_max, j_max)
        self.assertAlmostEqual(y[i_max], z_max, places=15)
