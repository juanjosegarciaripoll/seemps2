import numpy as np
from seemps.state import MPS
from ..tools import TestCase, random_uniform_mps, random_mps


class TestRandomUniformMPSStates(TestCase):
    def test_random_uniform_mps_produces_mps(self):
        mps = random_uniform_mps(2, 3)
        self.assertIsInstance(mps, MPS)

    def test_random_uniform_mps_size_value(self):
        self.assertEqual(len(random_uniform_mps(2, 3)), 3)
        self.assertEqual(len(random_uniform_mps(2, 10)), 10)

    def test_random_uniform_mps_dimensions_not_truncated(self):
        mps = random_uniform_mps(2, 3, D=1, truncate=False)
        shapes = [A.shape for A in mps]
        self.assertEqual(shapes, [(1, 2, 1), (1, 2, 1), (1, 2, 1)])

        mps = random_uniform_mps(2, 3, D=2, truncate=False)
        shapes = [A.shape for A in mps]
        self.assertEqual(shapes, [(1, 2, 2), (2, 2, 2), (2, 2, 1)])

        mps = random_uniform_mps(2, 3, D=3, truncate=False)
        shapes = [A.shape for A in mps]
        self.assertEqual(shapes, [(1, 2, 3), (3, 2, 3), (3, 2, 1)])

    def test_random_uniform_mps_complex_numbers(self):
        mps = random_uniform_mps(2, 3, D=1, complex=False)
        self.assertTrue(all(A.dtype == np.float64 for A in mps))

        mps = random_uniform_mps(2, 3, D=1, complex=True)
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


class TestRandomMPSStates(TestCase):
    def test_random_mps_produces_mps(self):
        mps = random_mps([2, 3, 2])
        self.assertIsInstance(mps, MPS)

    def test_random_mps_dimensions(self):
        d1 = [2, 3, 2, 5]
        mps = random_mps(d1)
        d2 = mps.physical_dimensions()
        self.assertEqual(d1, d2)

    def test_random_mps_size_value(self):
        self.assertEqual(len(random_mps([2, 3, 4])), 3)
        self.assertEqual(len(random_mps([3] * 10)), 10)

    def test_random_mps_dimensions_not_truncated(self):
        mps = random_mps([2, 3, 2], D=1, truncate=False)
        shapes = [A.shape for A in mps]
        self.assertEqual(shapes, [(1, 2, 1), (1, 3, 1), (1, 2, 1)])

        mps = random_mps([2, 3, 2], D=2, truncate=False)
        shapes = [A.shape for A in mps]
        self.assertEqual(shapes, [(1, 2, 2), (2, 3, 2), (2, 2, 1)])

        mps = random_mps([2, 3, 2], D=3, truncate=False)
        shapes = [A.shape for A in mps]
        self.assertEqual(shapes, [(1, 2, 3), (3, 3, 3), (3, 2, 1)])

    def test_random_mps_complex_numbers(self):
        mps = random_mps([2, 3, 2], D=1, complex=False)
        self.assertTrue(all(A.dtype == np.float64 for A in mps))

        mps = random_mps([2, 3, 2], D=1, complex=True)
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
