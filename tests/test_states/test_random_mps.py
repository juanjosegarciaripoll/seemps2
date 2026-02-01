import numpy as np
from seemps.state import MPS, random_uniform_mps, random_mps
from ..tools import SeeMPSTestCase


class TestRandomUniformMPSStates(SeeMPSTestCase):
    def test_random_uniform_mps_produces_mps(self):
        mps = random_uniform_mps(2, 3, rng=self.rng)
        self.assertIsInstance(mps, MPS)

    def test_random_uniform_mps_size_value(self):
        self.assertEqual(len(random_uniform_mps(2, 3, rng=self.rng)), 3)
        self.assertEqual(len(random_uniform_mps(2, 10, rng=self.rng)), 10)

    def test_random_uniform_mps_dimensions_not_truncated(self):
        mps = random_uniform_mps(2, 3, D=1, truncate=False, rng=self.rng)
        shapes = [A.shape for A in mps]
        self.assertEqual(shapes, [(1, 2, 1), (1, 2, 1), (1, 2, 1)])

        mps = random_uniform_mps(2, 3, D=2, truncate=False, rng=self.rng)
        shapes = [A.shape for A in mps]
        self.assertEqual(shapes, [(1, 2, 2), (2, 2, 2), (2, 2, 1)])

        mps = random_uniform_mps(2, 3, D=3, truncate=False, rng=self.rng)
        shapes = [A.shape for A in mps]
        self.assertEqual(shapes, [(1, 2, 3), (3, 2, 3), (3, 2, 1)])

    def test_random_uniform_mps_complex_numbers(self):
        mps = random_uniform_mps(2, 3, D=1, complex=False, rng=self.rng)
        self.assertTrue(all(A.dtype == np.float64 for A in mps))

        mps = random_uniform_mps(2, 3, D=1, complex=True, rng=self.rng)
        self.assertTrue(all(A.dtype == np.complex128 for A in mps))

    def test_random_uniform_mps_uses_rng(self):
        rng1 = np.random.default_rng(seed=0x1231)
        rng2 = np.random.default_rng(seed=0x1231)
        self.assertTrue(
            all(
                np.all(A == B)
                for A, B in zip(
                    random_uniform_mps(2, 10, rng=rng1),
                    random_uniform_mps(2, 10, rng=rng2),
                )
            )
        )


class TestRandomMPSStates(SeeMPSTestCase):
    def test_random_mps_produces_mps(self):
        mps = random_mps([2, 3, 2], rng=self.rng)
        self.assertIsInstance(mps, MPS)

    def test_random_mps_dimensions(self):
        d1 = [2, 3, 2, 5]
        mps = random_mps(d1, rng=self.rng)
        d2 = mps.physical_dimensions()
        self.assertEqual(d1, d2)

    def test_random_mps_size_value(self):
        self.assertEqual(len(random_mps([2, 3, 4], rng=self.rng)), 3)
        self.assertEqual(len(random_mps([3] * 10, rng=self.rng)), 10)

    def test_random_mps_dimensions_not_truncated(self):
        mps = random_mps([2, 3, 2], D=1, truncate=False, rng=self.rng)
        shapes = [A.shape for A in mps]
        self.assertEqual(shapes, [(1, 2, 1), (1, 3, 1), (1, 2, 1)])

        mps = random_mps([2, 3, 2], D=2, truncate=False, rng=self.rng)
        shapes = [A.shape for A in mps]
        self.assertEqual(shapes, [(1, 2, 2), (2, 3, 2), (2, 2, 1)])

        mps = random_mps([2, 3, 2], D=3, truncate=False, rng=self.rng)
        shapes = [A.shape for A in mps]
        self.assertEqual(shapes, [(1, 2, 3), (3, 3, 3), (3, 2, 1)])

    def test_random_mps_complex_numbers(self):
        mps = random_mps([2, 3, 2], D=1, complex=False, rng=self.rng)
        self.assertTrue(all(A.dtype == np.float64 for A in mps))

        mps = random_mps([2, 3, 2], D=1, complex=True, rng=self.rng)
        self.assertTrue(all(A.dtype == np.complex128 for A in mps))

    def test_random_mps_uses_rng(self):
        rng1 = np.random.default_rng(seed=0x1231)
        rng2 = np.random.default_rng(seed=0x1231)
        self.assertTrue(
            all(
                np.all(A == B)
                for A, B in zip(
                    random_mps([2] * 10, rng=rng1),
                    random_mps([2] * 10, rng=rng2),
                )
            )
        )
